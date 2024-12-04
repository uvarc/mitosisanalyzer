import glob
import cv2
import os
import numpy as np
import pandas as pd

from nd2reader import ND2Reader
from typing import Tuple
from tifffile import imread, TiffFile


def get_files(path, fpattern="*.tif") -> list:
    """Find files in a directory matching a customizable file name pattern"""
    files = []
    if os.path.isfile(path):
        files = [path]
    elif os.path.isdir(path):
        for pattern in fpattern:
            files.extend(glob.glob(os.path.join(path, pattern)))
        # remove possible duplicates and sort
        files = list(set(files))
        files.sort()
    return files


def save_movie(images, fname, codec="mp4v"):
    """Saves multi-dimensional numpy array as movie file."""
    height, width, _ = images[0].shape
    fourcc = cv2.VideoWriter_fourcc(*codec)
    vout = cv2.VideoWriter()
    success = vout.open(fname, fourcc, 15, (width, height), True)
    for img in images:
        vout.write(img)
    vout.release()
    return success


def nd2_opener(fname) -> Tuple[np.array, dict]:
    metadata = {}
    with ND2Reader(fname) as imagestack:
        pixelres = imagestack.metadata["pixel_microns"]
        metadata["axes"] = ["t", "c", "y", "x"]  # imagestack.axes
        metadata["pixel_unit"] = "um"
        metadata["pixel_res"] = pixelres
        metadata["scale"] = (
            f"{1/metadata['pixel_res']} pixels per {metadata['pixel_unit']}"
        )
        # set order tcyx and convert to np array
        try:
            imagestack.bundle_axes = "cyx"
            imagestack.iter_axes = "t"
            imagestack = np.array(imagestack)
        except:
            imagestack.bundle_axes = "yx"
            imagestack.iter_axes = "t"
            imagestack = np.array(imagestack)
            imagestack = np.expand_dims(imagestack, axis=1)
        metadata["shape"] = {
            metadata["axes"][i]: imagestack.shape[i]
            for i in range(len(imagestack.shape))
        }
    print(f"metadata['shape']={metadata['shape']}")
    print(f"imagestack.shape={imagestack.shape}")
    return imagestack, metadata


def tif_opener(fname) -> Tuple[np.array, dict]:
    imagestack = imread(fname)
    tif = TiffFile(fname)
    tags = tif.pages[0].tags
    # for t in [t for t in tags if "ij" not in str(t).lower()]:
    #    print (type(t), t, t.value)
    metadata = {}
    unit = RES_UNIT_DICT[tags["ResolutionUnit"].value]
    axesorder = [o.lower() for o in tif.series[0].axes]
    metadata["axes"] = axesorder
    metadata["shape"] = {axesorder[i]: s for i, s in enumerate(imagestack.shape)}
    metadata["pixel_unit"] = "um"
    pixelres = tags["XResolution"].value[1] / tags["XResolution"].value[0]
    if unit == "cm":
        metadata["pixel_res"] = pixelres * 10000
    elif unit == "inch":
        metadata["pixel_res"] = pixelres * 25400
    else:
        metadata["pixel_res"] = pixelres
    metadata["scale"] = (
        f"{tags['XResolution'].value[0]/tags['XResolution'].value[1]} pixels per {unit}"
    )
    return imagestack, metadata


def skip_opener(fname) -> Tuple[np.array, dict]:
    return None, None


def get_opener(fname):  # -> Tuple[np.array, dict]:
    ext = os.path.splitext(fname)[1]
    if ext == ".nd2":
        return nd2_opener
    elif ext in [".tif", ".tiff"]:
        return tif_opener
    return skip_opener
