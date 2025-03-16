import cv2
import dask.array as da
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from math import atan2, cos, sin, sqrt, pi
from prefect import task
from prefect.logging import get_run_logger
from typing import Optional, Union

from .calc import get_angle, center, euclidian
from .image import rotate_image, crop_stack, adjust_intensity


def draw_lines(
    img: np.ndarray,
    lines: list[list[tuple[int, int]]],
    color: Optional[tuple[int, int, int]] = (255, 255, 255),
    thickness: Optional[int] = 1,
    copy: Optional[bool] = False,
) -> None:
    """Draws a group of lines to image using specified color and line thickness.

    Parameters
        ----------
        img : np.ndarray
            Represents the 2-dimensional image.
        lines : list of list of tuples
            Defines list of lines, each with a start and end point.
            Example: [[(0,0),(512,512)]] defines line from (0,0) to (512,512).
        color : int, optional
            Sets line color in RGB (R, G, B).
        thickness: int, optional
            Sets the line thickness.
        copy: bool, optional
            If True, draws on a copy of the image. If False, the input image will be modified. Default: False
    """
    if copy:
        img = np.copy(img)
    for l in lines:
        img = cv2.line(img, l[0], l[1], color, thickness)
    return img


@task
def kymograph(
    images: np.ndarray,
    allpoles: np.ndarray,
    time_axis: Optional[int] = 0,
    width: Optional[int] = 200,
    height: Optional[int] = 10,
    adjust: Optional[bool] = False,
    method: Optional[str] = "sum",
    pad: Optional[int] = None,
):
    """Creates a kymgraph from an TCYX image stack based on the linescan through two points.
    The position for the two reference points has to be defined for each image plane.

    Parameters
        ----------
        images: np.ndarray
            The image stack. Expected TCYX
        allpoles: np.ndarray
            2-dimensional array of tuples with a pair spindle pole coordinates for each plane [[(x1, y1), (x2, y2)], ...]
        time_axis: int
            The index of the time axis in the image array. This is the axis to be projected.
        width: int, optional
            The length (in pixels) of the linescan throught the spindle pole coordinates. Default: 200.
        height: int, optional
            The width (in pixel) of the linescan, i.e orthogonal to the linescan direction. Default: 10.
        adjust: bool, optional
            If True, adjust the intensity in each linescan before creating kymograph. Default: False
        method: str, optional
            The method to project the width of the linescan box to a singl epixel width linescan. Default: sum, alternative max.
        pad: int, optional
            Pixels to pad around the images before processing. Default: None.

    Returns:
        kymo: np.ndarray
            The kymograph for the entire image stack.
        cropped_rot: np.ndarray
            Stack of the rotated images cropped to specified width and height such that the two spindle poles in each plane are aligned along the horizontal axis.
    """
    logger = get_run_logger()
    if pad is not None:
        images = [np.pad(img, pad) for img in images]
    allangles = np.array([get_angle(pole1, pole2) for (pole1, pole2) in allpoles])
    allcenters = [center(pole1, pole2) for (pole1, pole2) in allpoles]

    # Create linescan through spindle pole coordinates of specified width and height.
    # The center is the midpoint between the spindle poles
    rotated_images = np.array(
        [
            rotate_image(images[i], allangles[i], center=allcenters[i])
            for i in range(len(images))  # passing CYX per loop iteration
        ]
    )
    cropped_rot = crop_stack(rotated_images, width, height, center=True)
    logger.info(f"cropped_rot.shape={cropped_rot.shape}")
    if adjust:
        # stretch adjustment of intensity (YX plane) individually across all higher level TC axes
        planes = (-1, cropped_rot.shape[-2], cropped_rot.shape[-1])
        cropped_rot = adjust_intensity(
            cropped_rot,
            mode="stretch",
            axes=(-2, -1),  # Y and X are the last two axes in array
            clip_limit=0.0,
        )

    # Create projection of linescan y-axis so that height->1.
    # The time axis of the projected linescan becomes the y-axis in the kymograph so the y and time axis need to be swapped.
    # Squeeze out the 1-dimensionsional y-axis in the kymograph
    y_axis = -2
    if method == "sum":
        kymo = np.swapaxes(
            np.sum(cropped_rot, axis=y_axis), time_axis, y_axis
        ).squeeze()
    else:
        kymo = np.swapaxes(
            np.max(cropped_rot, axis=y_axis), time_axis, y_axis
        ).squeeze()
    # stretch contrast
    kymo = cv2.normalize(
        np.array(kymo, dtype=kymo.dtype), None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U
    )
    print(f"kymo.shape={kymo.shape}, cropped_rot.shape={cropped_rot.shape}")
    return kymo  # , cropped_rot


@task
def enhanced_kymograph(kymo, allpoles, relthr=0.7, padding=10):
    """Creates a stylized kymgraph from a CYX kymograph.

    Parameters
        ----------
        kymo: np.ndarray
            A 2-dimensional array representing the kymograph.
        allpoles: np.ndarray
            2-dimensional array of tuples with a pair of spindle pole coordinates for each plane [[(x1, y1), (x2, y2)], ...]
        relthr: float, optional
            Relative threshold to apply to the dna channel of the kymograph. Default: 0.7.
        padding: int, optional
            Pixels to pad around the images before processing. Default: 10.

    Returns:
        ekymo: np.ndarray
            The kymograph for the entire image stack.
    """
    poledistances = np.array(
        [euclidian(p1=pole1, p2=pole2) for (pole1, pole2) in allpoles]
    )
    dna_kymo = kymo[:, :, 2]
    width = dna_kymo.shape[1]
    dna_kymo = cv2.medianBlur(dna_kymo, 3, 0)
    thresh = cv2.adaptiveThreshold(
        dna_kymo, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 21, 7
    )  # need to rescale 0-1 for skeletoonize
    for row in range(len(kymo)):
        thresh[row, 0 : int(0.5 * (width + padding - poledistances[row]))] = 0
        thresh[row, int(0.5 * (width - padding + poledistances[row])) : -1] = 0
    ekymo = cv2.distanceTransform(thresh, cv2.DIST_L2, 3).astype(np.uint8)
    ekymo = cv2.normalize(ekymo, None, 0, 255.0, norm_type=cv2.NORM_MINMAX)
    return ekymo


def create_plots(
    df: pd.DataFrame, pixel_unit: Optional[str] = "?", plot_dna: Optional[bool] = True
) -> matplotlib.figure.Figure:
    """Plots a multi-panel figure with the spindle pole and DNA data over time.

    Parameters
        ----------
        df: pd.DataFrame
            The dataframe with the timeseries data.
        pixel_units: str, optional
            The unit of distance measurements.
        plot_dna: bool, optional
            If True, adds panels with DNA velocity and distance from midzone measurements.

    Returns
        ----------
        fig: plt.figure.Figure
            A Matplotlib figure.
    """
    if plot_dna:
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 6))
        # plot DNA position
        ax1.axhline(y=0.0, ls="-", color="tab:gray", lw=0.5)
        sns.lineplot(
            ax=ax1,
            palette=["tab:blue", "tab:orange"],
            data=df[
                [
                    f"left DNA-Midzone dist ({pixel_unit})",
                    f"right DNA-Midzone dist ({pixel_unit})",
                ]
            ],
        )
        ax1.set_title("Chromatid distance from midzone")
        ax1.set_ylabel(f"Distance ({pixel_unit})")
        ax1.set_xlim(left=0)
        for t in ax1.get_legend().texts:
            t.set_text(t.get_text().split(" ")[0])
        # plot DNA velocity
        ax2.axhline(y=0.0, ls="-", color="tab:gray", lw=0.5)
        sns.lineplot(
            ax=ax2,
            palette=["tab:blue", "tab:orange"],
            data=df[
                [
                    f"left DNA velocity ({pixel_unit}/frame)",
                    f"right DNA velocity ({pixel_unit}/frame)",
                ]
            ],
        )
        ax2.set_title("Chromatid velocity relative to midzone ")
        ax2.set_ylabel(f"Velocity ({pixel_unit}/frame)")
        ax2.set_xlim(left=0)
        for t in ax2.get_legend().texts:
            t.set_text(t.get_text().split(" ")[0])
    else:
        fig, ax3 = plt.subplots(1, 1, figsize=(10, 2))
    # plot pole oscillations velocity
    ax3.axhline(y=0.0, ls="-", color="tab:gray", lw=0.5)
    sns.lineplot(
        ax=ax3,
        palette=["tab:blue", "tab:orange"],
        data=df[[f"Pole 1 Osc ({pixel_unit})", f"Pole 2 Osc ({pixel_unit})"]],
    )
    ax3.set_title("Pole oscillations relative to embryo long axis")
    ax3.set_ylabel(f"Displacement ({pixel_unit})")
    ax3.set_xlim(left=0)
    for t in ax3.get_legend().texts:
        t.set_text(t.get_text().split(" ")[:2])
    # avoid overlap of plots
    fig.tight_layout()
    return fig


def drawAxis(
    img: np.ndarray,
    p_: tuple[int, int],
    q_: tuple[int, int],
    color: tuple[int, int, int],
    scale: float,
) -> None:
    """Draws an arrow from point p_ through q_. The length of the arrow can be scaled by factor <scale>.

    Parameters
        ----------
        img: np.ndarray
            Image to draw the arrow on.
        p_: tuple
            Defines starting point (x,y) of arrow.
        q_: tuple
            Defines endpoint (x,y) of arrow.
        color: tuple[int, int, int]
            Defines the RGB values of the arrow.
        scale: float
            Scales the length of the arrow symmetrically beyond p_ and q_.
    """
    p = list(p_)
    q = list(q_)

    ## [visualization1]
    angle = atan2(p[1] - q[1], p[0] - q[0])  # angle in radians
    hypotenuse = sqrt((p[1] - q[1]) * (p[1] - q[1]) + (p[0] - q[0]) * (p[0] - q[0]))

    # Here we lengthen the arrow by a factor of scale
    q[0] = p[0] - scale * hypotenuse * cos(angle)
    q[1] = p[1] - scale * hypotenuse * sin(angle)
    cv2.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), color, 3, cv2.LINE_AA)

    # create the arrow hooks
    p[0] = q[0] + 9 * cos(angle + pi / 4)
    p[1] = q[1] + 9 * sin(angle + pi / 4)
    cv2.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), color, 3, cv2.LINE_AA)

    p[0] = q[0] + 9 * cos(angle - pi / 4)
    p[1] = q[1] + 9 * sin(angle - pi / 4)
    cv2.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), color, 3, cv2.LINE_AA)
    ## [visualization1]
