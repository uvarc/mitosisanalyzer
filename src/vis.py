import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from math import atan2, cos, sin, sqrt, pi

from calc import get_angle, center, euclidian
from segmentation import rotate_image


def crop_stack(imagestack, width, height):
    # print (imagestack.shape)
    x1 = int((imagestack[0].shape[1] - width) / 2)
    y1 = int((imagestack[0].shape[0] - height) / 2)
    x2 = x1 + width
    y2 = y1 + height
    cropped = np.array([img[y1:y2, x1:x2] for img in imagestack])
    return cropped


def kymograph(images, allpoles, width=200, height=10, method="sum", pad=None):
    if pad is not None:
        images = [np.pad(img, pad) for img in images]
    print(f"input image format for kymograph: {images[0].shape}")
    allangles = np.array([get_angle(pole1, pole2) for (pole1, pole2) in allpoles])
    allcenters = [center(pole1, pole2) for (pole1, pole2) in allpoles]
    # print (np.median(allangles), np.min(allangles), np.max(allangles))
    rotated_images = [
        rotate_image(images[i], allangles[i], center=allcenters[i])
        for i in range(len(images))
    ]
    cropped_rot = crop_stack(rotated_images, width, height)
    if method == "sum":
        kymo = np.array([np.sum(img, axis=0) for img in cropped_rot])
    else:
        kymo = np.array([np.max(img, axis=0) for img in cropped_rot])
    kymo = cv2.normalize(kymo, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    print(f"kymo.shape={kymo.shape}, cropped_rot.shape={cropped_rot.shape}")
    return kymo, cropped_rot


def enhanced_kymograph(kymo, allpoles, relthr=0.7, padding=10):
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


def create_plots(df, pixel_unit="?", plot_dna=True):
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


def drawAxis(img, p_, q_, color, scale):
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
