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
from prefect import task, flow, Flow, unmapped
from prefect_dask import DaskTaskRunner
from prefect.task_runners import ConcurrentTaskRunner

from .calc import *
from .fileutils import *
from .segmentation import *
from .vis import *

EMBRYO_DIAM_UM = 50  # in micron


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
    #    parser.add_argument(
    #        "-k",
    #        "--kymograph",
    #        default=200,
    #        type=int,
    #        help="The value determines the width in pixel of the kymograph plots. Default 200. A value of 0 omits creation of kymographs.",
    #    )
    parser.add_argument(
        "-c",
        "--cellpose",
        action=argparse.BooleanOptionalAction,  # "store_true",
        help="use Cellpose to detect embryo contour",
    )
    parser.add_argument(
        "-f",
        "--framerate",
        default=None,
        type=float,
        help="number of frames per second",
    )
    parser.add_argument(
        "-e",
        "--executor",
        default="sequential",
        type=str,
        help="set executor. Options: sequential, concurrent, dask",
    )
    return parser


def validate_args(args):
    return args.blur % 2 == 1


def draw_lines(img, lines, color=(255, 255, 255), thickness=1):
    """Draws a group of lines to image using specified color and line thickness"""
    for l in lines:
        img = cv2.line(img, l[0], l[1], color, thickness)
    return img


def register_stack(imagestack):
    return imagestack


def process_spindle(image, polesize=20, blur=9, offset=(0, 0), threshold=0.0):
    height, width = image.shape
    # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    # image = clahe.apply(image)
    if blur > 0:
        blurredimg = cv2.GaussianBlur(image, (blur, blur), 0)
    else:
        blurredimg = image.copy()
    hist, bins = np.histogram(blurredimg.ravel(), 256, [0, 256])

    # for low in range(255, 0, -1):
    #    ret1, binary = cv2.threshold(blurredimg, low, 255, cv2.THRESH_BINARY)
    #    print(f"threshold={low}, binary.mean={np.mean(binary)}")

    if threshold == 0.0:
        ret1, th_img = cv2.threshold(
            blurredimg, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
    else:
        ret1, th_img = cv2.threshold(
            blurredimg, int(255 * threshold), 255, cv2.THRESH_BINARY
        )
        # th_img = cv2.adaptiveThreshold(
        #     blurredimg, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, blur, 2
        # )

    sp_contours, hierarchy = cv2.findContours(
        th_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
    )
    # print(f"spindle contours found: {len(sp_contours)}")
    if len(sp_contours) == 0:
        return (
            image,
            th_img,
            th_img,
            np.array([(0, 0), (0, 0)]),
            np.zeros(8).reshape(4, 2),
        )
    # print (f'type(sp_contours)={type(sp_contours)}')
    # print (f'type(sp_contours[0])={type(sp_contours[0])}')
    sp_contours = sorted(sp_contours, key=cv2.contourArea, reverse=True)

    # if len(sp_contours) > 2:
    #    sp_contours = sp_contours[:2]

    # for c in sp_contours:
    #    print (f'\tarea={cv2.contourArea(c)}')
    #   print (f'\tcenter={get_centers(c)}')
    binary = np.zeros((height, width), np.uint8)
    cv2.fillPoly(binary, sp_contours, (255, 255, 255))
    poles = [get_centers(c) for c in sp_contours]
    if len(sp_contours) > 1:
        # connect all contours with line to turn into single object
        color = (255, 255, 255)
        thickness = 2
        for i in range(len(sp_contours) - 1):
            binary = cv2.line(binary, poles[i], poles[i + 1], color, thickness)
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
        if threshold == 0.0:
            ret1, tmpbinary = cv2.threshold(
                img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
        else:
            ret1, tmpbinary = cv2.threshold(
                img, int(threshold * 255), 255, cv2.THRESH_BINARY
            )
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
    return image, th_img, binary, poles, corners  # spindle_poles

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
    binary = np.zeros((height, width), np.uint8)
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
    offset=(0, 0),
):
    # reshape and replace 0/0 coordinates with nan
    polearray = np.array(allpoles).reshape(len(allpoles), 4)
    polearray = np.where(polearray <= 0, np.nan, polearray)

    # shift pole and center coordinates by offset
    polearray[:, 0] = polearray[:, 0] + offset[0]  # add x oofset
    polearray[:, 1] = polearray[:, 1] + offset[1]  # add y offset
    polearray[:, 2] = polearray[:, 2] + offset[0]  # add x offset
    polearray[:, 3] = polearray[:, 3] + offset[1]  # add y offset
    center = (center[0] + offset[0], center[1] + offset[1])

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

    # calculate midzone position: gemoetric center between the two poles
    df["Midzone,x (pixel)"] = 0.5 * (df["Pole 2,x (pixel)"] + df["Pole 1,x (pixel)"])
    df["Midzone,y (pixel)"] = 0.5 * (df["Pole 2,y (pixel)"] + df["Pole 1,y (pixel)"])

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
    print(f"Pole-pole distance: mean={mean}, median={median}, std={std}")
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
    else:
        refline = (refline[0] + offset[0], refline[1] + offset[1])
    print(f"ref line={refline}")

    # refline_norm = np.linalg.norm(refline[1] - refline[0])

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
    kymo_width=200,
    do_plots=True,
):
    if dna_ch < 1:
        print(f"Processing: {fname}, spindle channel:{spindle_ch}, no dna channel")
    else:
        print(
            f"Processing: {fname}, spindle channel:{spindle_ch}, dna channel:{dna_ch}"
        )
    opener = get_opener(fname)
    imagestack, metadata = opener(fname)
    if output:
        fname = os.path.join(output, os.path.split(fname)[1])
    print(f"\t{opener}")
    for k, v in metadata.items():
        print(f"\t{k}:{v}")
    if max(spindle_ch, dna_ch) > metadata["shape"]["c"]:
        print("Skipping -- not enough channels.")
        return

    imagestack = register_stack(imagestack)

    width = metadata["shape"]["x"]  # imagestack.sizes['x']
    height = metadata["shape"]["y"]  # imagestack.sizes['y']

    if dna_ch > 0:
        dna_stack = np.array(imagestack)[:, dna_ch - 1]
    spindle_stack = np.array(imagestack)[:, spindle_ch - 1]
    embryo_masks = find_embryos(
        spindle_stack,
        threshold=threshold,
        cellpose=cellpose,
        cellpose_diam=int(EMBRYO_DIAM_UM / metadata["pixel_res"]),
    )
    print(f"Identified {len(embryo_masks)} embryos.")

    for embryo_no, emask in enumerate(embryo_masks):
        contours, _ = cv2.findContours(emask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        x, y, w, h = cv2.boundingRect(contours[0])
        emask_cropped, crop_origin, crop_width, crop_height = crop_image(
            emask, crop_origin=(x, y), crop_width=w, crop_height=h, copy=True
        )
        print(f"x={x}, y={y}, w={w}, h={h}")
        """
        cv2.imwrite(
            os.path.join(output, f"{embryo_no+1}-emask_cropped.png"), emask_cropped
        )
        cv2.imwrite(os.path.join(output, f"{embryo_no+1}-emask.png"), emask)
        """
        blank = np.zeros((crop_height, crop_width), np.uint8)
        blank_full = np.zeros((height, width), np.uint8)

        print(
            f"embryo {embryo_no+1}: len(contours)={len(contours)}, offset={crop_origin}"
        )
        for i, c in enumerate(contours):
            # Calculate the area of each contour
            area = cv2.contourArea(c)
            print(f"embryo_no {embryo_no+1}, area {i+1}: {area}")
            # Ignore contours that are too small or too large
            if area < min_area or area > max_area:
                continue
            # Draw each contour only for visualisation purposes
            # cv.drawContours(img, contours, i, (0, 0, 255), 2)
            # Find the orientation of each shape
            else:
                embryo_angle, embryo_center, embryo_norm, refline = get_orientation(
                    c, emask_cropped
                )
                print(f"embryo_norm={embryo_norm}")
                break

        allpoles = []
        allcorners = []
        allchromatids = []
        spimages = []

        print(f"Applying blur={blur} to find spindle poles")
        for frame_no, spimg in enumerate(spindle_stack):
            # normalize and convert to 8-bit
            spimg = cv2.normalize(spimg, None, 0, 255.0, norm_type=cv2.NORM_MINMAX)
            spimg = np.uint8(spimg)
            spimages.append(spimg)
            # crop to embryo bounding box
            spimg, _, _, _ = crop_image(
                spimg,
                crop_origin=crop_origin,
                crop_width=crop_width,
                crop_height=crop_height,
            )

            spimg = cv2.bitwise_and(spimg, spimg, mask=emask_cropped)
            spimg = cv2.normalize(spimg, None, 0, 255.0, norm_type=cv2.NORM_MINMAX)
            spimg, th_img, binary, spindle_poles, corners = process_spindle(
                spimg, blur=blur, threshold=threshold
            )
            """
            cv2.imwrite(
                os.path.join(output, f"{embryo_no+1}-{frame_no}-th_img.png"),
                th_img,
            )
            cv2.imwrite(
                os.path.join(output, f"{embryo_no+1}-{frame_no}-emask.png"),
                emask_cropped,
            )
            cv2.imwrite(
                os.path.join(output, f"{embryo_no+1}-{frame_no}-spimg.png"), spimg
            )
            cv2.imwrite(
                os.path.join(output, f"{embryo_no+1}-{frame_no}-binary.png"), binary
            )
            """
            allpoles.append(spindle_poles)  # (end1,end2)
            allcorners.append(corners)

        if dna_ch > 0:
            dnaimages = []
            dnabinimages = []
            for frame_no, dnaimg in enumerate(dna_stack):
                # normalize and convert to 8-bit
                dnaimg = cv2.normalize(dnaimg, None, 0, 255, norm_type=cv2.NORM_MINMAX)
                dnaimg = np.uint8(dnaimg)
                dnaimages.append(dnaimg)
                # crop to embryo bounding box
                dnaimg, _, _, _ = crop_image(
                    dnaimg,
                    crop_origin=crop_origin,
                    crop_width=crop_width,
                    crop_height=crop_height,
                )

                dnaimg = cv2.bitwise_and(dnaimg, dnaimg, mask=emask_cropped)
                dnaimg = cv2.normalize(dnaimg, None, 0, 255, norm_type=cv2.NORM_MINMAX)
                dnaimg, binary, chromatids = process_dna(dnaimg)
                dnabinimages.append(cv2.merge([blank, blank, binary]))
                chromatids = [c for c in chromatids if emask_cropped[c[1]][c[0]] == 255]
                allchromatids.append(chromatids)

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
            offset=crop_origin,
        )

        pole_dist = [euclidian(p1=p1, p2=p2) for (p1, p2) in fixed_poles]
        pole_dist_nan = len(pole_dist) - np.count_nonzero(~np.isnan(pole_dist))
        if pole_dist_nan > 0:
            print(f"Count of NaN values in pole_dist: {pole_dist_nan}")
            print(f"Aborting processing of embryo_no {embryo_no+1}")
            continue
        # print(pole_dist)
        left_pole = [int(0.5 * (kymo_width - d)) for d in pole_dist]
        # left_pole = [np.nan] * len(pole_dist)
        right_pole = [int(0.5 * (kymo_width + d)) for d in pole_dist]
        # right_pole = [np.nan] * len(pole_dist)

        df["left Pole (pixel)"] = left_pole
        df["right Pole (pixel)"] = right_pole

        if dna_ch > 0:
            # images = [
            #    cv2.merge([blank, spimages[i], dnaimages[i]])
            #    for i in range(len(spimages))
            # ]
            images = [
                cv2.merge(
                    [
                        blank_full,
                        # (imagestack[i, spindle_ch - 1] / 256).astype(np.uint8),
                        spimages[i],
                        # (imagestack[i, dna_ch - 1] / 256).astype(np.uint8),
                        dnaimages[i],
                    ]
                )
                for i in range(len(spimages))
            ]

            if kymo_width > 0:
                padding = (
                    (
                        crop_origin[1],
                        height - crop_origin[1] - crop_height,
                    ),
                    (
                        crop_origin[0],
                        width - crop_origin[0] - crop_width,
                    ),
                    (0, 0),
                )
                kymo, cropped_images = kymograph(
                    images,
                    fixed_poles,
                    width=kymo_width,
                    height=25,
                    method="max",
                    pad=None,
                )
                print(f"dnabinimages[0] format={dnabinimages[0].shape}")
                dnakymo, cropped_dna = kymograph(
                    dnabinimages,
                    fixed_poles,
                    width=kymo_width,
                    height=25,
                    method="max",
                    pad=padding,
                )
                print(f"dnakymo.shape={dnakymo.shape}")
                for i, line in enumerate(dnakymo):
                    dnakymo[i, left_pole[i], 1] = 255
                    dnakymo[i, right_pole[i], 1] = 255
                gaussian = 3
                # dnakymo = cv2.GaussianBlur(dnakymo, (gaussian, gaussian), 0)
                dnakymo = cv2.normalize(
                    dnakymo, None, 0, 255.0, norm_type=cv2.NORM_MINMAX
                )
                ekymo = enhanced_kymograph(kymo, fixed_poles, padding=15)

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
            for i, frame_chromatids in enumerate(allchromatids):
                for c in frame_chromatids:
                    images[i] = cv2.circle(images[i], (c[0], c[1]), 4, (255, 255, 0), 1)
        else:
            # images = [
            #    cv2.merge([blank, spimages[i], blank]) for i in range(len(spimages))
            # ]
            images = [
                cv2.merge(
                    [
                        blank_full,
                        # (imagestack[i, spindle_ch - 1] / 256).astype(np.uint8),
                        spimages[i],
                        blank_full,
                    ]
                )
                for i in range(len(imagestack))
            ]
        if kymo_width > 0:
            kymo, cropped_images = kymograph(
                images, fixed_poles, width=kymo_width, height=25, method="max"
            )
            ekymo = enhanced_kymograph(kymo, fixed_poles, padding=15)
            kymofile = (
                os.path.splitext(fname)[0]
                + f"-embryo-{(embryo_no+1):04d}-kymo-blur{(blur):02d}.png"
            )
            ekymofile = (
                os.path.splitext(fname)[0]
                + f"-embryo-{(embryo_no+1):04d}-ekymo-blur{(blur):02d}.png"
            )
            print(
                "Saving kymographs",
                kymofile,
                ekymofile,
            )
            cv2.imwrite(kymofile, kymo)
            cv2.imwrite(ekymofile, ekymo)
            if dna_ch > 0:
                dnakymofile = (
                    os.path.splitext(fname)[0]
                    + f"-embryo-{(embryo_no+1):04d}-dnakymo.png"
                )
                cv2.imwrite(dnakymofile, dnakymo)

        for i, frame_poles in enumerate(fixed_poles):
            for p in frame_poles:
                images[i] = cv2.circle(
                    images[i], (int(p[0]), int(p[1])), 4, (255, 0, 255), 1
                )
            # drawAxis(images[i], refline[0], refline[1], (127, 127, 0), 1)

        print(f"Processed embryo {embryo_no+1}")
        if valid:
            datafile = (
                os.path.splitext(fname)[0]
                + f"-embryo-{(embryo_no+1):04d}-blur{(blur):02d}.csv"
            )
            moviefile = os.path.splitext(fname)[0] + f"-embryo-{(embryo_no+1):04d}.mp4"
            cropped_moviefile = (
                os.path.splitext(fname)[0] + f"-embryo-{(embryo_no+1):04d}-cropped.mp4"
            )

            embryo_maskfile = (
                os.path.splitext(fname)[0]
                + f"-embryo-{(embryo_no+1):04d}-mask-blur.png"
            )
            vel_plotfile = (
                os.path.splitext(fname)[0] + f"-embryo-{(embryo_no+1):04d}-velocity.png"
            )
            df.to_csv(datafile)
            save_movie(images, moviefile)
            save_movie(cropped_images, cropped_moviefile)
            # save_movie(rotated_images, os.path.splitext(fname)[0] + '-rot.mp4')
            cv2.imwrite(embryo_maskfile, emask)
            if dna_ch > 0:
                dna_moviefile = (
                    os.path.splitext(fname)[0] + f"-embryo-{(embryo_no+1):04d}-dna.mp4"
                )
                save_movie(dnabinimages, dna_moviefile)

            if do_plots:
                fig = create_plots(
                    df, pixel_unit=metadata["pixel_unit"], plot_dna=dna_ch > 0
                )
                fig.savefig(vel_plotfile)

            print(f"Saved embryo {embryo_no+1}")
    # except:
    # print (f"Error extracting features from embryo {embryo_no} -- skipping")


@task
def proc_file(fname, spindle_ch, dna_ch, output):
    print(f"Processing {fname}")
    import time

    time.sleep(10)
    return f"Processed {fname}, {spindle_ch}, {dna_ch}, {output}"


@task
def print_file(f):
    print(f)


@flow(
    name="Analysis",
    log_prints=True,
)
def run(args=None):
    """Main code block"""
    files = get_files.submit(args.input, fpattern=["*.nd2", "*.tiff", "*.tif"])
    print_file.map(files).wait()
    processed = process_file.map(
        files,
        spindle_ch=unmapped(args.spindle),
        dna_ch=unmapped(args.dna),
        output=unmapped(args.output),
        refframe=unmapped(args.refframe),
        threshold=unmapped(args.threshold),
        blur=unmapped(args.blur),
        cellpose=unmapped(args.cellpose),
        kymo_width=unmapped(200),  # args.kymograph),
    ).wait()
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
    # flow.run()


@flow
def configured_flow(args=None):  # pass this to `Deployment.build_from_flow`
    if args.executor == "concurrent":
        task_runner = ConcurrentTaskRunner()
    elif args.executor == "dask":
        task_runner = DaskTaskRunner(
            cluster_kwargs={
                "n_workers": 8,
                "threads_per_worker": 1,
                "resources": {"GPU": 1, "process": 1},
            }
        )
    else:
        task_runner = None
    return run.with_options(task_runner=task_runner)(args=args)


def main():
    parser = init_parser()
    args = parser.parse_args()
    if not validate_args(args):
        print("Illegal arguments. Blur requires an odd value 1-15.")
    else:
        #    with Flow(
        #        name="Analysis",
        #        task_runner=DaskTaskRunner(
        #            cluster_kwargs={"n_workers": 4, "resources": {"GPU": 1, "process": 1}}
        #        ),
        #    ) as flow:
        configured_flow(args=args)


if __name__ == "__main__":
    main()
