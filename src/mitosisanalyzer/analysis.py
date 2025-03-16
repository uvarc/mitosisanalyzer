import os
import argparse
import glob
import cv2
import numpy as np
import pandas as pd
import math
import time

from bioio_base.dimensions import DEFAULT_CHUNK_DIMS
from math import atan2, cos, sin, sqrt, pi
from typing import Optional, Any
from prefect import task, flow, Flow, unmapped, context
from prefect_dask import DaskTaskRunner
from prefect.task_runners import ConcurrentTaskRunner, TaskRunner
from prefect.logging import get_run_logger
from prefect.context import get_run_context
from prefect.deployments import run_deployment

from .calc import *
from .utils import *
from .image import *
from .vis import *

CELL_DIAM_UM = 50  # in micron
UNKNOWN_UNIT = "unknown"


def init_parser() -> argparse.ArgumentParser:
    """Parses command line arguments.

    Returns
        ----------
        parser: argparse.ArgumentParser
    """
    parser = argparse.ArgumentParser(
        description="Analyzes spindle pole and chromatid movements in .nd2 timelapse files"
    )
    parser.add_argument(
        "-i",
        "--input",
        required=True,
        help="image file or directory with image files to be processed",
    )
    parser.add_argument("-o", "--output", default=None, help="output file or directory")
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
        help="reference frame to determine spindle pole axis (0=autodetect based on cell long axis)",
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
        action=argparse.BooleanOptionalAction,
        help="use Cellpose to detect cell contour",
    )
    parser.add_argument(
        "--kymograph",
        action=argparse.BooleanOptionalAction,
        help="create kymographs",
    )
    parser.add_argument(
        "--plots",
        action=argparse.BooleanOptionalAction,
        help="create spindle pole velocity plots",
    )
    parser.add_argument(
        "--movies",
        action=argparse.BooleanOptionalAction,
        help="create movies with spindle pole overlays",
    )
    parser.add_argument(
        "-f",
        "--framerate",
        default=15.0,
        type=float,
        help="number of frames per second for created movies. Only used in combination with --movies",
    )
    parser.add_argument(
        "-e",
        "--executor",
        default="sequential",
        type=str,
        help="set executor. Options: sequential, concurrent, dask",
    )
    parser.add_argument(
        "--address",
        default="local",
        type=str,
        help="provide address to existing Dask Scheduler. Default: local, spins up a new Dask scheduler and clients on localhost using the --process and --threads options.",
    )
    parser.add_argument(
        "--processes",
        default=1,
        type=int,
        help="number of parallel processes. Ignored when --adress is set.",
    )
    parser.add_argument(
        "--threads",
        default=4,
        type=int,
        help="number of threads per process. Ignored when --address is set.",
    )
    return parser


def validate_args(args: argparse.Namespace) -> bool:
    """Validates the command line arguments.

    Parameters
        ----------
        args : argparse.Namespace object
            Contains the individual command line arguments.

    Returns
        ----------
        valid : boolean
            returns True if all tests for individual command line arguments return True.
    """
    # blur value has to be an odd number >=1
    valid = args.blur > 0 and args.blur % 2 == 1
    return valid


def process_spindle(
    image: np.ndarray,
    polesize: Optional[int] = 20,
    blur: Optional[int] = 9,
    threshold: Optional[float] = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[tuple]]:
    """
    Determines on a single channel image plane the (x,y) position of the spindle poles. An optional Gaussian blur is applied before thresholding the image for segementation.

    Parameters
        ----------
        image: np.ndarray
            Represents a single channel 2-dimensional image.
        polesize: int, optional
            Option to set max pole diameter (pixel). Not used.
        blur: int, optional
            The width and height of the Gaussian blur kernel. It has to be an odd number >0. This step is skipped if the blur value is <1.
        offset: tuple, optional
            (x,y) pixel offset to add to the center of the segmented spindle poles.
            This is useful when processing cropped images and the spindle pole coordinates
            should reflect the position in the ucropped image. The default is (0,0).
        threshold: float, optional
            Value 0.0 <= threshold <= 1.0. For a value of 0.0 the Otsu methods is applied.

    Returns
        ----------
        image: np.ndarray
           The input image.
        th_img; np.ndarray
           The thresholded image (8-bit)
        binary: np.ndarray
           The binary image (8-bit)
        poles: list of tuples
           Each tuple defines a 2-dimensional spindle pole coordinate (x, y)
        corners: list of tuples
           The four tuples define the corners of each spindle poles bounding boxes
    """
    height, width = image.shape
    # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    # image = clahe.apply(image)

    # apply blur (optional)
    if blur > 0:
        blurredimg = cv2.GaussianBlur(image, (blur, blur), 0)
    else:
        blurredimg = image.copy()
    hist, bins = np.histogram(blurredimg.ravel(), 256, [0, 256])

    # apply threshold
    if threshold == 0.0:
        ret1, th_img = cv2.threshold(
            blurredimg, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
    else:
        ret1, th_img = cv2.threshold(
            blurredimg, int(255 * threshold), 255, cv2.THRESH_BINARY
        )

    # identify contours on thresholded image and sort them by area in descending order
    sp_contours, hierarchy = cv2.findContours(
        th_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
    )
    if len(sp_contours) == 0:
        return (
            image,
            th_img,
            th_img,
            np.array([(0, 0), (0, 0)]),
            np.zeros(8).reshape(4, 2),
        )
    sp_contours = sorted(sp_contours, key=cv2.contourArea, reverse=True)

    # fill all contours and define them as single contour
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

    # get corners of bounding box and
    corners = get_rect_points(sp_contours[0])
    edges = get_edges(corners)
    edges, scorners = square_edges(edges[:2])

    #
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
            cv2.fillPoly(binary, pts=sp_contours, color=(255, 255, 255))
            poles.append(get_centers(sp_contours[0]))
        else:
            poles.append([np.nan, np.nan])

    return image, th_img, binary, np.array(poles), corners  # spindle_poles


def process_dna(image: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[tuple]]:
    """
    Performs segmentation on a single channel image plane to determine the (x,y) center position of chromatids.

    Parameters
        ----------
        image: np.ndarray
           Represents a single channel 2-dimensional image (YX).

    Returns
        ----------
        image: np.ndarray
           A single focal plane image (YX).
        binary: np.ndarray
           The binary image (8-bit)
        chromatids: list of tuples
           Each tuple defines a chromatids center as (x, y) coordinate.
    """
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


@task
def create_dataframe(
    allpoles: np.ndarray,
    # allchromatids,
    center: tuple[int, int],
    refframe: int,
    refline: Optional[list[tuple[int, int]]] = None,
    pixel_res: Optional[float] = 1.0,
    pixel_unit: Optional[str] = "um",
    framerate: Optional[float] = 1.0,
    frame_unit: Optional[str] = UNKNOWN_UNIT,
    rolling: Optional[int] = 3,
    offset: Optional[tuple[int, int]] = (0, 0),
) -> pd.DataFrame:
    """
    Creates a pd.DataFrame object from the spindle coordinates.

    Parameters
        ----------
        allpoles: np.ndarray
            The assumption is that the array is Tx[(pole1_x,pole1-y),(pole_2x, pole_2y)]
        center: tuple
            Center of the cell in pixel coordinates as (x, y)
        refframe: int
            Index in most outer axis (T) in allpoles that defines the spindle axis that will be used
            to calculate the pole oscillation amplitudes over time.
        refline: list of 2-dimentional tuples, optional
            If this is not None, the refframe parameter will be ignored and the spindle pole oscillations
            will be calculated relative to the axis defined by line through two points defined as
            P1=(refline[0][0], refline[0][1]) and P2=(refline[1][0], refline[1][1]).
        pixel_res: float, optional
            defines the physica dimension of a pixel in realworld units.
        pixel_unit: str, optional
            The unit of the pixel_res value.
        rolling: int, optional
            Applies a rolling average over the rows of the dataframe.
        offset: tuple, optional
            (x,y) pixel offset to add to the center of the segmented spindle poles.
            This is useful when processing cropped images and the spindle pole coordinates
            should reflect the position in the ucropped image. The default is (0,0).

    Returns
        ----------
        df: pd.DataFrame
    """
    # adding offset and reshape into 2d array
    polearray = (allpoles + offset).reshape((allpoles.shape[0], 4))
    print(f"polearray.shape={polearray.shape}")
    polearray = np.where(polearray <= 0, np.nan, polearray)

    # shift pole and center coordinates by offset
    # polearray[:] = polearray[:] + np.array(offset).reshape((2,))
    # polearray[:, 0] = polearray[:, 0] + offset[0]  # add x offset
    # polearray[:, 1] = polearray[:, 1] + offset[1]  # add y offset
    # polearray[:, 2] = polearray[:, 2] + offset[0]  # add x offset
    # polearray[:, 3] = polearray[:, 3] + offset[1]  # add y offset
    center = center + offset

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

    df["Cell center (pixel)"] = f"({center[0]}/{center[1]})"
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

    df["Frame"] = np.arange(1, len(allpoles) + 1)
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

    # MOVE REFLINE CALCUlATION
    if refline is None:
        # p = next(x for x in allpoles if not isnan(x))
        cntr = (
            int(0.5 * (df.iloc[refframe - 1, 0] + df.iloc[refframe - 1, 2])),
            int(0.5 * (df.iloc[refframe - 1, 1] + df.iloc[refframe - 1, 3])),
        )
        deltax = df.iloc[refframe - 1, 0] - df.iloc[refframe - 1, 2]
        deltay = df.iloc[refframe - 1, 1] - df.iloc[refframe - 1, 3]
        first_norm = np.array([deltax, deltay])
        refline = (cntr + first_norm * 10, cntr - first_norm * 10)
    else:
        refline = (refline[0] + offset[0], refline[1] + offset[1])
    print(f"ref line={refline}")
    # END OF REFLINE CALCULATION

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

    allpoles = df.iloc[:, 0:4].values
    allpoles = allpoles.reshape((len(allpoles), 2, 2))

    return df, allpoles, True, refline  # valid


@task
def process_file(
    fname: str,
    spindle_ch: int,
    dna_ch: int,
    output: str,
    refframe: Optional[int] = 0,
    min_area: Optional[int] = 3700,
    max_area: Optional[int] = 100000,
    threshold: Optional[float] = 0.0,
    blur: Optional[float] = 9.0,
    cellpose: Optional[bool] = False,
    kymo_width: Optional[int] = 200,
    with_plots: Optional[bool] = True,
):
    """Reads and processes the images in a given file.

    Parameters
        ----------
        fname: str
            The file to read and process. Can include a relative or absolute path.
        spindle_ch
            The channel index that contains the spindle pole images. First channel is 1, not 0.
        dna_ch
            The channel index that contains the dna images. First channel is 1, not 0. If set to 0,
            analysis of chromatid positions will be skipped even if such channel exists in the raw data.
        output: str
            Directory to save all output file to. If not specified, files will be saved in the directory
            of the input image file.
        refframe: int, optional
            Index in most outer axis (T) in allpoles that defines the spindle axis that will be used
            to calculate the pole oscillation amplitudes over time.
        min_area: int, optional
            Not used.
        max_area: int, optional
            Not used.
        threshold: float, optional
            Value 0.0 <= threshold <= 1.0. For a value of 0.0 the Otsu methods is applied. Default: 0.0.
        blur: int, optional
            The width and height of the Gaussian blur kernel that is applied before cell segmentation.
            It has to be an odd number >0. This step is skipped if the blur value is <1. Default: 9.
        cellpose: boolean, optional
            If True, the Cellpose Cyto3 model will be used to segment the cells. Default: False.
        kymo_width: int, optional
            Length of linescan in pixel which will become the width of the resulting kymograph. Default: 200.
        with_plot: boolean: optional
            If True, creates and saves plots in output directory. Default: True.
    """
    logger = get_run_logger()
    if dna_ch < 1:
        logger.info(f"Reading {fname}, spindle channel:{spindle_ch}, no dna channel")
    else:
        logger.info(
            f"Reading {fname}, spindle channel:{spindle_ch}, dna channel:{dna_ch}"
        )
    # set up file opener
    opener = get_opener(fname)
    logger.debug(f"\t{opener}")

    # read image file with meta data
    imagestack, metadata = opener(fname)
    width = metadata["shape"]["x"]  # imagestack.sizes['x']
    height = metadata["shape"]["y"]  # imagestack.sizes['y']
    for k, v in metadata.items():
        logger.debug(f"\t{k}:{v}")

    if max(spindle_ch, dna_ch) > metadata["shape"]["c"]:
        logger.info(f"Skipping {fname} -- not enough channels.")
        return

    # register image stack
    imagestack = register_stack(imagestack)

    # create separate stacks of spindle channel images and dna channel images
    if dna_ch > 0:
        dna_stack = np.array(imagestack)[:, dna_ch - 1]
    spindle_stack = np.array(imagestack)[:, spindle_ch - 1]

    cell_masks = find_cells(
        spindle_stack,
        threshold=threshold,
        cellpose=cellpose,
        cellpose_diam=int(CELL_DIAM_UM / metadata["pixel_res"]),
    )
    logger.info(f"Identified {len(cell_masks)} cells.")

    for embryo_no, emask in enumerate(cell_masks):
        # find largest contour
        contours, _ = cv2.findContours(emask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        x, y, w, h = cv2.boundingRect(contours[0])
        logger.debug(f"x={x}, y={y}, w={w}, h={h}")

        # crop cell mask
        emask_cropped, crop_origin, crop_width, crop_height = crop_image(
            emask, crop_origin=(x, y), crop_width=w, crop_height=h, copy=True
        )

        logger.info(
            f"cell {embryo_no+1}: len(contours)={len(contours)}, offset={crop_origin}"
        )
        for i, contour in enumerate(contours):
            # Calculate the area of each contour
            area = cv2.contourArea(contour)
            logger.info(f"embryo_no {embryo_no+1}, area {i+1}: {area}")
            # Ignore contours that are too small or too large
            if area < min_area or area > max_area:
                continue
            # Draw each contour only for visualisation purposes
            # cv.drawContours(img, contours, i, (0, 0, 255), 2)
            # Find the orientation of each shape
            else:
                embryo_angle, embryo_center, embryo_norm, refline = get_orientation(
                    contour
                )
                logger.debug(f"embryo_norm={embryo_norm}")
                break

        allpoles = []
        allcorners = []
        allchromatids = []
        spimages = []

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

            allpoles.append(spindle_poles)  # (end1,end2)
            allcorners.append(corners)

        blank = np.zeros((crop_height, crop_width), np.uint8)
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
            logger.info(f"Count of NaN values in pole_dist: {pole_dist_nan}")
            logger.info(f"Aborting processing of embryo_no {embryo_no+1}")
            continue
        # print(pole_dist)
        left_pole = [int(0.5 * (kymo_width - d)) for d in pole_dist]
        # left_pole = [np.nan] * len(pole_dist)
        right_pole = [int(0.5 * (kymo_width + d)) for d in pole_dist]
        # right_pole = [np.nan] * len(pole_dist)

        df["left Pole (pixel)"] = left_pole
        df["right Pole (pixel)"] = right_pole

        blank_full = np.zeros((height, width), np.uint8)
        if dna_ch > 0:
            images = [
                cv2.merge(
                    [
                        blank_full,
                        spimages[i],
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
            images = [
                cv2.merge(
                    [
                        blank_full,
                        spimages[i],
                        blank_full,
                    ]
                )
                for i in range(len(imagestack))
            ]

        # set path and file basename for all output files
        if output:
            fname = os.path.join(output, os.path.split(fname)[1])

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

            if with_plots:
                fig = create_plots(
                    df, pixel_unit=metadata["pixel_unit"], plot_dna=dna_ch > 0
                )
                fig.savefig(vel_plotfile)

            print(f"Saved embryo {embryo_no+1}")
    # except:
    # print (f"Error extracting features from embryo {embryo_no} -- skipping")


@task
def reduce_results(results: list[tuple[Any, ...]]) -> tuple[list[Any], ...]:
    """Reduces the results from mapped tasks into separate lists."""
    return tuple(zip(*results))


@task
def process_spindle_stack(
    spindle_stack: np.ndarray,
    cellmask: np.ndarray,
    blur: Optional[int] = 9,
    threshold: Optional[float] = 0.5,
) -> np.ndarray:
    logger = get_run_logger()
    logger.info(
        f"Locating spindle poles: type(spindle_stack)={type(spindle_stack)}, spindle_stack.shape={spindle_stack.shape}, cellmask.shape={cellmask.shape}"
    )
    allpoles = np.empty((spindle_stack.shape[0], 2, 2))  # no_frames * [[x1,y1],[x,y2]]
    allpoles[:] = np.nan
    # allcorners = []
    # spimages = []

    for frame_no, spimg in enumerate(spindle_stack):
        # mask image with cell_mask and find spindle poles
        # spimg = cv2.normalize(
        #    np.array(spimg), None, 0, 255.0, norm_type=cv2.NORM_MINMAX
        # )
        spimg, th_img, binary, spindle_poles, corners = process_spindle(
            np.array(spimg, dtype=spimg.dtype), blur=blur, threshold=threshold
        )
        allpoles[frame_no, :, :] = spindle_poles
        # allcorners.append(corners)

    return allpoles


@task
def process_dna_stack(
    dna_stack: np.ndarray,
    cellmask: np.ndarray,
    crop_origin: Optional[tuple[int, int]] = (0, 0),
):
    logger = get_run_logger()
    logger.info(
        f"Locating DNA: dna_stack.shape={dna_stack.shape}, cellmask.shape={cellmask.shape}"
    )
    allchromatids = []
    # blank = np.zeros(cellmask.shape, np.uint8)
    # dnaimages = []
    # dnabinimages = []
    for frame_no, dnaimg in enumerate(dna_stack):
        # normalize and convert to 8-bit
        # dnaimg = cv2.normalize(dnaimg, None, 0, 255, norm_type=cv2.NORM_MINMAX)
        # dnaimg = np.uint8(dnaimg).squeeze()
        # dnaimages.append(dnaimg)

        ## crop to embryo bounding box
        # dnaimg, _, _, _ = crop_image(
        #    dnaimg,
        #    crop_origin=crop_origin,
        #    crop_width=crop_width,
        #    crop_height=crop_height,
        # )

        dnaimg = np.array(dnaimg, dtype=dnaimg.dtype)
        dnaimg = cv2.bitwise_and(dnaimg, dnaimg, mask=cellmask)
        dnaimg = cv2.normalize(dnaimg, None, 0, 255, norm_type=cv2.NORM_MINMAX)
        dnaimg, binary, chromatids = process_dna(dnaimg)
        # dnabinimages.append(cv2.merge([blank, blank, binary]))
        chromatids = [c for c in chromatids if cellmask[c[1]][c[0]] == 255]
        allchromatids.append(chromatids)


@task(name="Process Cell")
def process_cell(
    imagestack: Union[np.ndarray, da.Array],
    metadata: dict[str, Any],
    spindle_ch: int,
    dna_ch: int,
    cellmask: np.ndarray,
    cellid: int,
    fname: str,
    output: str,
    blur: Optional[int] = 9,
    threshold: Optional[float] = 0.5,
    refframe: Optional[int] = 0,
    with_kymograph: Optional[bool] = True,
    with_plots: Optional[bool] = False,
    with_movies: Optional[bool] = False,
    framerate: Optional[int] = 15,
    min_area: Optional[int] = None,
    max_area: Optional[int] = None,
) -> None:
    logger = get_run_logger()
    logger.info(
        f"Processing cell {cellid} in {fname}, type(imagestack={type(imagestack)}, imagestack.shape={imagestack.shape}, imagestack.dtype={imagestack.dtype}, blur={blur}, threshold={threshold}, refframe={refframe}"
    )
    future_results = []

    # find largest contour within min/max area range and crop cell mask
    contour = get_contour(cellmask, min_area=min_area, max_area=max_area)
    if contour is None:
        raise ValueError(f"no contour found with {min_area}<=area<={max_area}.")

    x, y, w, h = cv2.boundingRect(contour)
    cell_angle, cell_center, cell_norm, refline = get_orientation(contour)
    # crop mask
    cmask_cropped, crop_origin, crop_width, crop_height = crop_image(
        cellmask, crop_origin=(x, y), crop_width=w, crop_height=h, copy=True
    )

    # crop imagestack
    cropped_stack = crop_stack(
        imagestack,
        x=crop_origin[0],
        y=crop_origin[1],
        width=crop_width,
        height=crop_height,
    )

    # adjust intensity on cropped cell
    cropped_stack = adjust_intensity(
        cropped_stack, axes="YX", axes_labels=metadata["axes"], mode="stretch"
    )

    logger.info(
        f"cell_center={cell_center}, cell_angle={cell_angle}, cell_norm={cell_norm}, refline={refline}"
    )
    logger.info(
        f"Cropped: crop_origin={crop_origin}, cmask_cropped.shape={cmask_cropped.shape}, type(cropped_stack)={type(cropped_stack)}, cropped_stack.shape={cropped_stack.shape}))"
    )

    spindle_stack = cropped_stack[:, spindle_ch]
    tmp_poles = process_spindle_stack.submit(
        spindle_stack,
        cmask_cropped,
        blur=blur,
        threshold=threshold,
    )

    # store pole data in dataframe and save it
    if refframe > 0:
        refline = None
    df, fixed_poles, valid, refline = create_dataframe(
        tmp_poles,
        # allchromatids,
        cell_center,
        refframe,
        refline=refline,
        pixel_res=metadata["pixel_res"],
        pixel_unit=metadata["pixel_unit"],
        offset=crop_origin,  # make sure we translate pole coordinates for final output
    )

    basename = os.path.splitext(os.path.basename(fname))[0]
    df_filename = os.path.join(output, f"{basename}-cell-{(cellid):04d}.csv")
    logger.info(f"df_filename={df_filename}")
    df.to_csv(df_filename)

    pole_dist = [euclidian(p1=p1, p2=p2) for (p1, p2) in fixed_poles]
    pole_dist_nan = len(pole_dist) - np.count_nonzero(~np.isnan(pole_dist))
    if pole_dist_nan > 0:
        raise ValueError(
            f"Count of NaN values in pole_dist: {pole_dist_nan}. Aborting processing of {fname} cell {cellid}"
        )

    if dna_ch is not None:
        dna_stack = cropped_stack[:, dna_ch]
        dr = process_dna_stack.submit(dna_stack, cmask_cropped, crop_origin=crop_origin)
        future_results.append(dr)

    cropped_rgb = to_RGB.submit(
        cropped_stack,
        axes_labels=metadata["axes"],
        colors={"G": spindle_ch, "R": spindle_ch + 1, "B": 2},
    )

    mask_filename = os.path.join(output, f"{basename}-mask-cell-{(cellid):04d}.tif")
    cv2.imwrite(mask_filename, cellmask)

    if with_movies:
        movie_filename = os.path.join(output, f"{basename}-cell-{(cellid):04d}.mp4")
        logger.info(f"movie_filename={movie_filename}")
        saved = save_movie.submit(cropped_rgb, movie_filename, framerate=framerate)
        future_results.append(saved)

    if with_kymograph:
        kymo_width = 200
        # print(pole_dist)
        left_pole = [int(0.5 * (kymo_width - d)) for d in pole_dist]
        # left_pole = [np.nan] * len(pole_dist)
        right_pole = [int(0.5 * (kymo_width + d)) for d in pole_dist]
        # right_pole = [np.nan] * len(pole_dist)
        df["left Pole (pixel)"] = left_pole
        df["right Pole (pixel)"] = right_pole

        # full_merged = to_RGB.submit([imagestack[:, dna_ch], imagestack[:, spindle_ch]])
        kymo = kymograph.submit(
            imagestack,  # cropped_stack
            fixed_poles,  # - [crop_origin, crop_origin],
            time_axis=0,
            width=kymo_width,
            height=25,
            adjust=True,
            method="max",
        )
        kymo_rgb = to_RGB.submit([kymo.result()[dna_ch], kymo.result()[spindle_ch]])
        kymofile = os.path.join(
            output, f"{basename}-cell-{(cellid):04d}-kymo-blur{(blur):02d}.png"
        )
        cv2.imwrite(kymofile, kymo_rgb.result())

    if with_plots:
        vel_plotfile = os.path.join(
            output, f"{basename}-cell-{(cellid):04d}-velocity-blur{(blur):02d}.png"
        )
        logger.info(f"vel_plotfile={vel_plotfile}")
        fig = create_plots(df, pixel_unit=metadata["pixel_unit"], plot_dna=dna_ch > 0)
        fig.savefig(vel_plotfile)

    if dna_ch > 0:
        # dnakymofile = f"{os.path.splitext(fname)[0]}-cell-{(cellid):04d}-dnakymo.png"
        # cv2.imwrite(dnakymofile, dnakymo)
        pass

    # wait for all async tasks to finish
    [result.wait() for result in future_results]


# @flow(
#    name="Process spindles in FOV", log_prints=True, task_runner=ConcurrentTaskRunner()
# )
@task
def process_stack(
    imagestack: np.ndarray,
    metadata: dict[str, Any],
    cellmasks: list[np.ndarray],
    fname: str,
    output: str,
    spindle_ch: int,
    dna_ch: int,
    blur: Optional[int] = 9,
    threshold: Optional[float] = 0.5,
    refframe: Optional[int] = 1,
    with_kymograph: Optional[bool] = True,
    with_plots: Optional[bool] = False,
    with_movies: Optional[bool] = True,
    framerate: Optional[int] = 15,
) -> None:
    logger = get_run_logger()
    logger.info(
        f"Processing image stacks {imagestack.shape} for {fname} with {len(cellmasks)} cell(s)"
    )
    r = [
        process_cell.submit(
            **{
                "imagestack": imagestack,
                "metadata": metadata,
                "spindle_ch": spindle_ch,
                "dna_ch": dna_ch,
                "cellmask": cellmask,
                "cellid": cellid,
                "fname": fname,
                "output": output,
                "blur": blur,
                "threshold": threshold,
                "refframe": refframe,
                "with_kymograph": with_kymograph,
                "with_plots": with_plots,
                "with_movies": with_movies,
                "framerate": framerate,
            },
        )
        for cellid, cellmask in enumerate(cellmasks)
    ]
    logger.info(f"type(r)={type(r)}")
    import prefect

    prefect.futures.wait(r)


@task
def calc_celldiameter(metadata: dict[str:Any]) -> Any:
    """Converts distance in um to pixel units using metada["pixel_res"]"""
    return int(CELL_DIAM_UM / metadata["pixel_res"])


@task
def item_by_index(results: tuple[Any, ...], index: int = 0) -> Any:
    """Helper function to get item from iterable in mapped flow/task'"""
    return results[index]


@task
def item_by_key(data: dict[Any:Any], key: Any):
    """Helper function to get item from dictionary in mapped flow/task'"""
    return data[key]


@flow(
    name="Analyze files",
    log_prints=True,
    task_runner=ConcurrentTaskRunner(),
)
def run(
    input: str,
    spindle_ch: Optional[int] = 1,
    dna_ch: Optional[int] = 0,
    output: Optional[str] = ".",
    refframe: Optional[int] = 0,
    threshold: Optional[float] = 0.0,
    blur: Optional[int] = 9,
    cellpose: Optional[bool] = False,
    with_kymograph: Optional[bool] = True,
    with_plots: Optional[bool] = False,
    with_movies: Optional[bool] = True,
    framerate: Optional[int] = 15,
    backend: Optional[str] = None,
) -> None:
    """Get file list and process all files"""
    logger = get_run_logger()
    files = get_files(input, fpattern=["*.nd2", "*.tiff", "*.tif"])

    # read image file with metadata
    read_results = read_file.map(
        files, dask=backend == "dask", squeeze=True
    )  # , opener)
    # imagestacks = item_by_index.map(read_results, index=unmapped(0))
    # metadata = item_by_index.map(read_results, index=unmapped(1))

    imagestacks = [item_by_index.submit(r, index=0) for r in read_results]
    metadata = [item_by_index.submit(r, index=1) for r in read_results]
    axes = item_by_key.map(metadata, "axes")

    # register stacks
    rimagestacks = register_stack.map(imagestacks)

    # # adjust image intensity levels
    # imagestacks = adjust_intensity.map(
    #     imagestacks, unmapped("YX"), axes, unmapped("stretch")
    # )
    # # save adjusted stack as mp4 movie
    # rgb_stacks = to_RGB.map(
    #    rimagestacks, axes, unmapped({"G": spindle_ch, "R": spindle_ch + 1, "B": 2})
    # )
    # movie_filenames = [f"{os.path.splitext(fname)[0]}.mp4" for fname in files]
    # saved = save_movie.map(rgb_stacks, movie_filenames)
    # saved.wait()

    # split stack by channel
    spstacks = slice_stack.map(rimagestacks, axes, unmapped({"C": spindle_ch - 1}))
    spindlestacks = spstacks

    # calculate expected cell diameter in pixel units
    diameters = calc_celldiameter.map(metadata)

    cell_masks = find_cells.map(
        spindlestacks,
        threshold=unmapped(threshold),
        cell_diameter=diameters,
        cellpose=unmapped(cellpose),
    )
    stack_results = process_stack.map(
        rimagestacks,
        metadata,
        cell_masks,
        files,
        unmapped(output),
        unmapped(spindle_ch),
        unmapped(dna_ch),
        unmapped(blur),
        unmapped(threshold),
        unmapped(refframe),
        unmapped(with_kymograph),
        unmapped(with_plots),
        unmapped(with_movies),
        unmapped(framerate),
    )

    stack_results.wait()
    # dna_stacks.wait()


def select_task_runner(
    executor: Optional[str] = "concurrent",
    processes: Optional[int] = 1,
    threads: Optional[int] = 1,
    address: Optional[str] = None,
) -> tuple[TaskRunner, str]:
    """Selects the task runner for the flow."""
    task_runner = None
    task_runner_name = "Default Task Runner"
    backend = "loky"  # for joblib
    if executor == "concurrent":
        task_runner = ConcurrentTaskRunner()
        task_runner_name = f"{task_runner}"
    elif executor == "dask":
        print(f"address={address}")
        if address == "local":
            task_runner = DaskTaskRunner(
                performance_report_path=os.environ.get("DASK_PERFORMANCE_REPORT", None),
                cluster_kwargs={
                    "n_workers": processes,
                    "threads_per_worker": threads,
                    "resources": {"GPU": 1, "process": 1},
                },
            )
        else:
            task_runner = DaskTaskRunner(
                performance_report_path=os.environ.get("DASK_PERFORMANCE_REPORT", None),
                cluster=address,
            )
        task_runner_name = f"{task_runner}"
        backend = "dask"
    return task_runner, task_runner_name, backend


def main() -> None:
    parser = init_parser()
    args = parser.parse_args()
    if not validate_args(args):
        print("Illegal arguments. Blur requires an odd value 1-15.")
    else:
        # deploy the proces_cell so we can call it as subflow later
        # process_cell.serve(name="process-cell")
        task_runner, task_runner_name, backend = select_task_runner(
            executor=args.executor,
            processes=args.processes,
            threads=args.threads,
            address=args.address,
        )
        run.with_options(task_runner=task_runner)(
            input=args.input,
            spindle_ch=args.spindle - 1,
            dna_ch=args.dna - 1,
            output=args.output,
            refframe=args.refframe,
            threshold=args.threshold,
            blur=args.blur,
            cellpose=args.cellpose,
            with_kymograph=args.kymograph,
            with_plots=args.plots,
            with_movies=args.movies,
            framerate=args.framerate,
            backend=backend,
        )


if __name__ == "__main__":
    main()
