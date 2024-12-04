# -*- coding: utf-8 -*-'
#!/usr/bin/env python3
"""
Created on Wed Jul 14 21:42:03 2021

@author: khs3z
"""

import os
import argparse
import glob
import cv2
import numpy as np
import pandas as pd
import math

from math import atan2, cos, sin, sqrt, pi
from typing import Tuple
from prefect import task, Flow, unmapped

from calc import *
from fileutils import *
from segmentation import *
from vis import *

RES_UNIT_DICT = {1: "<unknown>", 2: "inch", 3: "cm"}


def init_parser():
    """Parses command line arguments"""
    parser = argparse.ArgumentParser(
        description="Analyzes spindle pole and chromatid movements in .nd2 timelapse files"
    )
    parser.add_argument(
        "-i",
        "--input",
        required=True,
        help=".nd2 file or directory with .nd2 files to be processed",
    )
    parser.add_argument("-o", "--output", default=None, help="output file or directory")
    parser.add_argument(
        "-p", "--processes", default=1, type=int, help="number or parallel processes"
    )
    parser.add_argument(
        "-s",
        "--spindle",
        default=1,
        type=int,
        help="channel # for tracking spindle poles",
    )
    parser.add_argument(
        "-d", "--dna", default=0, type=int, help="channel # for tracking dna"
    )
    parser.add_argument(
        "-r",
        "--refframe",
        default=0,
        type=int,
        help="reference frame to determine spindle pole axis (0=autodetect based on embryo long axis)",
    )
    parser.add_argument(
        "-t",
        "--threshold",
        default=0.0,
        type=float,
        help="threshold of cytoplasmic background signal in spindle channel; value relative to max spindle intensity 0.0-1.0 (0.0=autodetect using Otsu)",
    )
    parser.add_argument(
        "-b",
        "--blur",
        default=9,
        type=int,
        help="applies a gaussian blur before segmenting spindle poles. The value determines the blurring radius; a value of 0 omits blurring.",
    )
    parser.add_argument(
        "-c",
        "--cellpose",
        action="store_true",
        help="use Cellpose to detect embryo contour",
    )
    parser.add_argument(
        "-f",
        "--framerate",
        default=None,
        type=float,
        help="number of frames per second",
    )
    return parser


def draw_lines(img, lines, color=(255, 255, 255), thickness=1):
    """Draws a group of lines to image using specified color and line thickness"""
    for l in lines:
        img = cv2.line(img, l[0], l[1], color, thickness)
    return img


def register_stack(imagestack):
    return imagestack


def process_spindle(image, polesize=20, blur=9):
    height, width = image.shape
    # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    # image = clahe.apply(image)
    if blur > 0:
        blurredimg = cv2.GaussianBlur(image, (blur, blur), 0)
    else:
        blurredimg = image.copy()
    hist, bins = np.histogram(blurredimg.ravel(), 256, [0, 256])

    ret1, binary = cv2.threshold(blurredimg, 191, 255, cv2.THRESH_BINARY)

    sp_contours, hierarchy = cv2.findContours(
        binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
    )
    if len(sp_contours) == 0:
        return image, binary, np.array([(0, 0), (0, 0)]), np.zeros(8).reshape(4, 2)
    # print (f'type(sp_contours)={type(sp_contours)}')
    # print (f'type(sp_contours[0])={type(sp_contours[0])}')
    sp_contours = sorted(sp_contours, key=cv2.contourArea, reverse=True)
    if len(sp_contours) > 2:
        sp_contours = sp_contours[:2]
    # for c in sp_contours:
    #    print (f'\tarea={cv2.contourArea(c)}')
    #   print (f'\tcenter={get_centers(c)}')
    binary = np.zeros((height, width), np.uint8)
    cv2.fillPoly(binary, sp_contours, (255, 255, 255))
    poles = [get_centers(c) for c in sp_contours]
    if len(sp_contours) > 1:
        color = (255, 255, 255)
        thickness = 2
        binary = cv2.line(binary, poles[0], poles[1], color, thickness)
        sp_contours, hierarchy = cv2.findContours(binary, 1, 2)
    # print (f'\tlen(sp_contours): {len(sp_contours)}', flush=True)
    corners = get_rect_points(sp_contours[0])
    # print (f'\tcorners: {corners}', type(corners[0]))
    edges = get_edges(corners)
    # print (f'\tedges: {edges}')
    edges, scorners = square_edges(edges[:2])
    # print (f'\tsquare corners: {scorners}')
    poles = []
    binary = np.zeros((height, width), np.uint8)
    for square in scorners[:2]:
        mask = np.zeros((height, width), dtype=np.uint8)
        square = np.array(square).flatten().reshape((len(square), 2))
        # print (f'\tsquare={square}')
        cv2.fillPoly(mask, pts=[square], color=(255, 255, 255))
        img = cv2.bitwise_and(blurredimg, blurredimg, mask=mask)
        ret1, tmpbinary = cv2.threshold(img, 191, 255, cv2.THRESH_BINARY)
        sp_contours, hierarchy = cv2.findContours(
            tmpbinary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
        )
        if len(sp_contours) == 1:
            # print (f'scorners={scorners}, square={square}')
            # cv2.imshow('t',blurredimg)
            # cv2.waitKey(0)
            cv2.fillPoly(binary, pts=sp_contours, color=(255, 255, 255))
            poles.append(get_centers(sp_contours[0]))
        else:
            # cv2.fillPoly(binary, pts=sp_contours, color=(255,255,255))
            poles.append([np.nan, np.nan])

    # image = draw_lines(image,edges,color=(255,0,0),thickness=1)
    # print (f'\tspindle poles: {poles}')
    return image, binary, poles, corners  # spindle_poles

    # ret2,binary2 = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)


def process_dna(image):
    chromatids = []
    height, width = image.shape
    blurredimg = cv2.medianBlur(image, 3, 0)
    ret1, binary = cv2.threshold(blurredimg, 127, 255, cv2.THRESH_BINARY)
    # binary = cv2.adaptiveThreshold(blurredimg, 255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 10)

    dna_contours, hierarchy = cv2.findContours(
        binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
    )
    binary = np.zeros((width, height), np.uint8)
    cv2.fillPoly(binary, pts=dna_contours, color=(255, 255, 255))
    chromatids = [get_centers(c) for c in dna_contours]

    # dt, dna, binaries = watershed(cv2.merge([binary,binary,binary]),binary,dilate=1,erode=1,relthr=0.1)
    # chromatids = []
    # for b in binaries:
    #    dna_contours,hierarchy = cv2.findContours(b, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    #    chromatids.extend([get_centers(c) for c in dna_contours])
    ##print (chromatids)
    return image, binary, chromatids


def create_dataframe(
    allpoles,
    # allchromatids,
    center,
    refframe,
    refline=None,
    pixel_res=1.0,
    pixel_unit="um",
    rolling=3,
):
    # reshape and replace 0/0 coordinates with nan
    polearray = np.array(allpoles).reshape(len(allpoles), 4)
    polearray = np.where(polearray <= 0, np.nan, polearray)

    df = pd.DataFrame(
        polearray,
        columns=[
            "Pole 1,x (pixel)",
            "Pole 1,y (pixel)",
            "Pole 2,x (pixel)",
            "Pole 2,y (pixel)",
        ],
    )
    # df['Pole 1,x (pixel)'] = df['Pole 1,x (pixel)'].rolling(rolling).mean()
    # df['Pole 1,y (pixel)'] = df['Pole 1,y (pixel)'].rolling(rolling).mean()
    # df['Pole 2,x (pixel)'] = df['Pole 2,x (pixel)'].rolling(rolling).mean()
    # df['Pole 2,y (pixel)'] = df['Pole 2,y (pixel)'].rolling(rolling).mean()

    df["Embryo center (pixel)"] = f"({center[0]}/{center[1]})"
    # df['embryo angle'] = embryo_angle

    df["Midzone,x (pixel)"] = 0.5 * (df["Pole 2,x (pixel)"] + df["Pole 1,x (pixel)"])
    df["Midzone,y (pixel)"] = 0.5 * (df["Pole 2,y (pixel)"] + df["Pole 1,y (pixel)"])
    # find nan values, forward and backfill
    outliers = df.isna().any(axis=1)
    df = df.ffill()
    df = df.bfill()
    df["angle"] = df.apply(lambda row: get_row_angle(row), axis=1)
    med_angle = df["angle"].median()
    swap = (df["angle"] - med_angle).abs() > 90

    df.loc[swap, ["Pole 1,x (pixel)", "Pole 2,x (pixel)"]] = df.loc[
        swap, ["Pole 2,x (pixel)", "Pole 1,x (pixel)"]
    ].values
    df.loc[swap, ["Pole 1,y (pixel)", "Pole 2,y (pixel)"]] = df.loc[
        swap, ["Pole 2,y (pixel)", "Pole 1,y (pixel)"]
    ].values
    df["angle"] = df.apply(lambda row: get_row_angle(row), axis=1)

    # calculate pole distance
    df[f"Pole-Pole Distance [{pixel_unit}]"] = df.apply(
        lambda row: get_row_euclidian(row, pixel_res), axis=1
    )
    df["Pole 1 [pixel]"] = (
        "("
        + df["Pole 1,x (pixel)"].astype(str)
        + "/"
        + df["Pole 1,y (pixel)"].astype(str)
        + ")"
    )
    df["Pole 2 [pixel]"] = (
        "("
        + df["Pole 2,x (pixel)"].astype(str)
        + "/"
        + df["Pole 2,y (pixel)"].astype(str)
        + ")"
    )
    df["Midzone [pixel]"] = (
        "("
        + df["Midzone,x (pixel)"].astype(str)
        + "/"
        + df["Midzone,y (pixel)"].astype(str)
        + ")"
    )
    df["Frame"] = np.arange(1, len(allpoles) + 1)
    # df = df[['Frame', 'Pole 1 [pixel]', 'Pole 2 [pixel]', f'Pole-Pole Distance [{pixel_unit}]']]
    df = df.set_index("Frame")
    mean = df[f"Pole-Pole Distance [{pixel_unit}]"].mean()
    median = df[f"Pole-Pole Distance [{pixel_unit}]"].median()
    std = df[f"Pole-Pole Distance [{pixel_unit}]"].std()
    print(f"mean={mean}, median={median}, std={std}")
    valid = (
        median != 0.0
        and mean > 3.0
        and mean / median > 0.8
        and mean / median < 1.2
        and std < 0.4 * mean
    )
    allpoles = df.iloc[:, 0:4].values

    if refline is None:
        # p = next(x for x in allpoles if not isnan(x))
        cntr = (
            int(0.5 * (df.iloc[refframe - 1, 0] + df.iloc[refframe - 1, 2])),
            int(0.5 * (df.iloc[refframe - 1, 1] + df.iloc[refframe - 1, 3])),
        )
        deltax = df.iloc[refframe - 1, 0] - df.iloc[refframe - 1, 2]
        deltay = df.iloc[refframe - 1, 1] - df.iloc[refframe - 1, 3]
        first_norm = np.array([deltax, deltay])
        """
        x = np.concatenate((df.iloc[:, 0], df.iloc[:, 2]), axis=0)
        xn = np.argwhere(~np.isnan(x))
        y = np.concatenate((df.iloc[:, 1], df.iloc[:, 3]), axis=0)
        yn = np.argwhere(~np.isnan(y))
        # xyn = x and y
        print(x)
        print(y)
        print(f"x.shape={x.shape}")
        print(f"y.shape={y.shape}")
        A = np.vstack([x, np.ones(len(x))]).T
        m, c = np.linalg.lstsq(A, y, rcond=None)[0]
        print(f"m={m}, c={c}")
        fig, ax = plt.subplots(1, 1)
        ax.plot(x, y, 'o', label='Original data', markersize=2)
        ax.plot(x, m*x + c, 'r', label='Fitted line')
        ax.legend()
        fig.savefig("regression.png")
        """
        refline = (cntr + first_norm * 10, cntr - first_norm * 10)
    print(f"ref line={refline}")

    refline_norm = np.linalg.norm(refline[1] - refline[0])

    # calc oscillations
    df[f"Pole 1 Osc ({pixel_unit})"] = oscillation(
        refline[0],
        refline[1],
        df[["Pole 1,x (pixel)", "Pole 1,y (pixel)"]].values,
        pixel_res=pixel_res,
    )
    df[f"Pole 2 Osc ({pixel_unit})"] = oscillation(
        refline[0],
        refline[1],
        df[["Pole 2,x (pixel)", "Pole 2,y (pixel)"]].values,
        pixel_res=pixel_res,
    )

    allpoles = allpoles.reshape((len(allpoles), 2, 2))

    return df, allpoles, True, refline  # valid


@task
def process_file(
    fname,
    spindle_ch,
    dna_ch,
    output,
    refframe=0,
    min_area=3700,
    max_area=100000,
    threshold=0.0,
    blur=9.0,
    cellpose=False,
):
    if dna_ch < 1:
        print(f"Processing: {fname}, spindle channel:{spindle_ch}, no dna channel")
    else:
        print(
            f"Processing: {fname}, spindle channel:{spindle_ch}, dna channel:{dna_ch}"
        )
    opener = get_opener(fname)
    imagestack, metadata = opener(fname)
    print(f"\t{opener}")
    for k, v in metadata.items():
        print(f"\t{k}:{v}")
    if max(spindle_ch, dna_ch) > metadata["shape"]["c"]:
        print("Skipping -- not enough channels.")
        return

    imagestack = register_stack(imagestack)

    width = metadata["shape"]["x"]  # imagestack.sizes['x']
    height = metadata["shape"]["y"]  # imagestack.sizes['y']

    spindle_stack = np.array(imagestack)[:, spindle_ch - 1]
    if dna_ch > 0:
        dna_stack = np.array(imagestack)[:, dna_ch - 1]
    embryo_masks = find_embryos(spindle_stack, threshold=threshold, cellpose=cellpose)
    print(f"len(embryo_masks)={len(embryo_masks)}")

    blank = np.zeros((height, width), np.uint8)

    for embryo_no, embryo in enumerate(embryo_masks):
        # try:
        embryo_copy = np.copy(embryo)
        contours, _ = cv2.findContours(embryo, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        print(f"len(contours)={len(contours)}")
        for i, c in enumerate(contours):
            # Calculate the area of each contour
            area = cv2.contourArea(c)
            print(f"embryo_no {embryo_no}, area {i}: {area}")
            # Ignore contours that are too small or too large
            if area < min_area or area > max_area:
                continue
            # Draw each contour only for visualisation purposes
            # cv.drawContours(img, contours, i, (0, 0, 255), 2)
            # Find the orientation of each shape
            else:
                embryo_angle, embryo_center, embryo_norm, refline = get_orientation(
                    c, embryo_copy
                )
                print(f"embryo_norm={embryo_norm}")
                break

        allpoles = []
        allcorners = []
        allchromatids = []
        spimages = []

        for frame_no, spimg in enumerate(spindle_stack):
            spimg = cv2.normalize(spimg, None, 0, 255.0, norm_type=cv2.NORM_MINMAX)
            spimg = np.uint8(spimg)
            spimages.append(spimg)  # spimg
            spimg = cv2.bitwise_and(spimg, spimg, mask=embryo)
            spimg = cv2.normalize(spimg, None, 0, 255.0, norm_type=cv2.NORM_MINMAX)
            spimg, binary, spindle_poles, corners = process_spindle(spimg, blur=blur)
            allpoles.append(spindle_poles)  # (end1,end2)
            allcorners.append(corners)

        if dna_ch > 0:
            dnaimages = []
            dnabinimages = []
            for frame_no, dnaimg in enumerate(dna_stack):
                dnaimg = cv2.normalize(dnaimg, None, 0, 255, norm_type=cv2.NORM_MINMAX)
                dnaimg = np.uint8(dnaimg)
                dnaimages.append(dnaimg)

                dnaimg = cv2.bitwise_and(dnaimg, dnaimg, mask=embryo)
                dnaimg = cv2.normalize(dnaimg, None, 0, 255, norm_type=cv2.NORM_MINMAX)
                dnaimg, binary, chromatids = process_dna(dnaimg)
                dnabinimages.append(cv2.merge([blank, blank, binary]))
                chromatids = [c for c in chromatids if embryo[c[0]][c[1]] == 255]
                allchromatids.append(chromatids)

        kymo_width = 200
        allpoles = np.array(allpoles)

        if refframe > 0:
            refline = None
        df, fixed_poles, valid, refline = create_dataframe(
            allpoles,
            # allchromatids,
            embryo_center,
            refframe,
            refline=refline,
            pixel_res=metadata["pixel_res"],
            pixel_unit=metadata["pixel_unit"],
        )

        pole_dist = [euclidian(p1=p1, p2=p2) for (p1, p2) in fixed_poles]
        left_pole = [int(0.5 * (kymo_width - d)) for d in pole_dist]
        right_pole = [int(0.5 * (kymo_width + d)) for d in pole_dist]

        df["left Pole (pixel)"] = left_pole
        df["right Pole (pixel)"] = right_pole

        if dna_ch > 0:
            images = [
                cv2.merge([blank, spimages[i], dnaimages[i]])
                for i in range(len(spimages))
            ]
            kymo, cropped_images = kymograph(
                images, fixed_poles, width=kymo_width, height=25, method="max"
            )
            dnakymo, cropped_dna = kymograph(
                dnabinimages, fixed_poles, width=kymo_width, height=25, method="max"
            )
            for i, line in enumerate(dnakymo):
                dnakymo[i, left_pole[i], 1] = 255
                dnakymo[i, right_pole[i], 1] = 255
            gaussian = 3
            dnakymo = cv2.GaussianBlur(dnakymo, (gaussian, gaussian), 0)
            dnakymo = cv2.normalize(dnakymo, None, 0, 255.0, norm_type=cv2.NORM_MINMAX)
            for i, frame_chromatids in enumerate(allchromatids):
                for c in frame_chromatids:
                    images[i] = cv2.circle(images[i], (c[0], c[1]), 4, (255, 255, 0), 1)
            df["left DNA edge (pixel)"] = [
                (
                    np.where(line[:, 2] > 127)[0][0]
                    if len(np.where(line[:, 2] > 127)[0]) > 0
                    else -1
                )
                for line in dnakymo
            ]
            df["right DNA edge (pixel)"] = [
                (
                    np.where(line[:, 2] > 127)[0][-1]
                    if len(np.where(line[:, 2] > 127)[0]) > 0
                    else -1
                )
                for line in dnakymo
            ]
            df[f'left DNA-Pole dist ({metadata["pixel_unit"]})'] = (
                df["left DNA edge (pixel)"] - df["left Pole (pixel)"]
            ) * metadata["pixel_res"]
            df[f'right DNA-Pole dist ({metadata["pixel_unit"]})'] = (
                df["right Pole (pixel)"] - df["right DNA edge (pixel)"]
            ) * metadata["pixel_res"]
            df[f'left DNA-Midzone dist ({metadata["pixel_unit"]})'] = (
                df["left DNA edge (pixel)"] - 0.5 * kymo.shape[1]
            ) * metadata["pixel_res"]
            df[f'right DNA-Midzone dist ({metadata["pixel_unit"]})'] = (
                df["right DNA edge (pixel)"] - 0.5 * kymo.shape[1]
            ) * metadata["pixel_res"]
            df[f'left DNA velocity ({metadata["pixel_unit"]}/frame)'] = (
                df[f'left DNA-Midzone dist ({metadata["pixel_unit"]})']
                .diff(periods=1)
                .rolling(7)
                .mean()
            )
            df[f'right DNA velocity ({metadata["pixel_unit"]}/frame)'] = (
                df[f'right DNA-Midzone dist ({metadata["pixel_unit"]})']
                .diff(periods=1)
                .rolling(7)
                .mean()
            )
        else:
            images = [
                cv2.merge([blank, spimages[i], blank]) for i in range(len(spimages))
            ]
            kymo, cropped_images = kymograph(
                images, fixed_poles, width=kymo_width, height=25, method="max"
            )
        ekymo = enhanced_kymograph(kymo, fixed_poles, padding=15)

        print(f"kymo.shape={kymo.shape}")
        print(f"ekymo.shape={ekymo.shape}")
        for i, frame_poles in enumerate(fixed_poles):
            for p in frame_poles:
                images[i] = cv2.circle(
                    images[i], (int(p[0]), int(p[1])), 4, (255, 0, 255), 1
                )
            # drawAxis(images[i], refline[0], refline[1], (127, 127, 0), 1)

        print(f"Processed embryo {embryo_no}")
        if valid:
            if output:
                fname = os.path.join(output, os.path.split(fname)[1])
            datafile = os.path.splitext(fname)[0] + f"-embryo-{(embryo_no+1):04d}.csv"
            moviefile = os.path.splitext(fname)[0] + f"-embryo-{(embryo_no+1):04d}.mp4"
            cropped_moviefile = (
                os.path.splitext(fname)[0] + f"-embryo-{(embryo_no+1):04d}-cropped.mp4"
            )
            kymofile = (
                os.path.splitext(fname)[0] + f"-embryo-{(embryo_no+1):04d}-kymo.png"
            )
            ekymofile = (
                os.path.splitext(fname)[0] + f"-embryo-{(embryo_no+1):04d}-ekymo.png"
            )
            embryo_maskfile = (
                os.path.splitext(fname)[0] + f"-embryo-{(embryo_no+1):04d}-mask.png"
            )
            vel_plotfile = (
                os.path.splitext(fname)[0] + f"-embryo-{(embryo_no+1):04d}-velocity.png"
            )
            df.to_csv(datafile)
            save_movie(images, moviefile)
            save_movie(cropped_images, cropped_moviefile)
            # save_movie(rotated_images, os.path.splitext(fname)[0] + '-rot.mp4')
            cv2.imwrite(embryo_maskfile, embryo_copy)
            cv2.imwrite(kymofile, kymo)
            cv2.imwrite(ekymofile, ekymo)
            if dna_ch > 0:
                dna_moviefile = (
                    os.path.splitext(fname)[0] + f"-embryo-{(embryo_no+1):04d}-dna.mp4"
                )
                dnakymofile = (
                    os.path.splitext(fname)[0]
                    + f"-embryo-{(embryo_no+1):04d}-dnakymo.png"
                )
                save_movie(dnabinimages, dna_moviefile)
                cv2.imwrite(dnakymofile, dnakymo)

            fig = create_plots(
                df, pixel_unit=metadata["pixel_unit"], plot_dna=dna_ch > 0
            )
            fig.savefig(vel_plotfile)

            print(f"Saved embryo {embryo_no}")
    # except:
    # print (f"Error extracting features from embryo {embryo_no} -- skipping")


@task
def proc_file(fname, spindle_ch, dna_ch, output):
    return f"Processed {fname}"


def main():
    """Main code block"""
    parser = init_parser()
    args = parser.parse_args()
    with Flow("Analysis") as flow:
        files = get_files(args.input, fpattern=["*.nd2", "*.tiff", "*.tif"])
        processed = process_file.map(
            files,
            spindle_ch=unmapped(args.spindle),
            dna_ch=unmapped(args.dna),
            output=unmapped(args.output),
            refframe=unmapped(args.refframe),
            threshold=unmapped(args.threshold),
            blur=unmapped(args.blur),
            cellpose=unmapped(args.cellpose),
        )
        # print (f'Processing: {fname}, spindle channel:{spindle_ch}, dna channel:{dna_ch}'
        # opener = get_opener.map(files)
        # imagestack, metadata = opener.map(files)
        """
        print (f'\t{opener}')
        for k,v in metadata.items():
            print (f'\t{k}:{v}')
        if max(spindle_ch, dna_ch) > metadata['shape']['c']:
            print ("Skipping -- not enough channels.")
            return
        #pixel_microns = imagestack.metadata['pixel_microns']

        imagestack = register_stack(imagestack)

        width = metadata['shape']['x'] #imagestack.sizes['x']
        height = metadata['shape']['y'] #imagestack.sizes['y']
    
        #imagestack.bundle_axes = 'cxy'
        #imagestack.iter_axes = 't'
    
    
        #max_spindle_int = np.amax(np.array(imagestack)[:,1])
        #max_dna_int = np.amax(np.array(imagestack)[:,0])

        spindle_stack = np.array(imagestack)[:,spindle_ch-1]
        dna_stack = np.array(imagestack)[:,dna_ch-1]
        embryo_masks = find_embryos(spindle_stack)
    
        blank = np.zeros((height,width), np.uint8)

		"""
    # flow.visualize()
    flow.run()


if __name__ == "__main__":
    main()
