import glob
import cv2
import os
import dask.array as da
import numpy as np
import pandas as pd
import dask
import dask.array as da

from bioio import BioImage
from bioio_base.dimensions import DEFAULT_CHUNK_DIMS
import bioio_ome_tiff
from nd2reader import ND2Reader
from tifffile import imread, TiffFile

from prefect import task
from prefect.logging import get_run_logger
from typing import Optional, Union, Any, Callable

RES_UNIT_DICT = {1: "<unknown>", 2: "inch", 3: "cm"}


def create_array(
    arraylike: Union[np.ndarray, list[np.ndarray], list[da.Array]],
    delayed_shape: tuple[int, ...],
    out_shape: tuple[int, ...] = None,
    dtype=None,
    backend: str = "loky",
) -> Union[np.ndarray, da.Array]:
    """Combines a collection of numpy or dask arrays into a single numpy or dask array.

    Parameters
        ----------
        arraylike: np.ndarray | list[np.ndarray] | list[da.Array]
            The collection of arrays to concatenate
        delayed_shape: tuple[int, ...]
            Defines the shape of the individual items in arraylike. only needed for dask arrays. Only used if
            `backend == "dask"`. Ignored for all other backends.
        out_shape: tuple[int, ...], optional
            The returned array wille be reshaped to match out_shape. Default: None, no reshaping.
        dtype:
            Convert returned array to specified dtype. Default: None, keep dtype of the input collection.
        backend: str, optional
            If `backend == "dask"`, each item in arraylike will wrapped as dask.delayed object and then
            concatenated into single dask array. The dask array will be persisted if the environment variable
            `DASK_PERSIST_INPUT` is set. A numpy array will be returned for all other backend descriptors.

    Returns
        ----------
        Numpy or dask array, depending on `backend` parameter value.
    """
    if dtype is None:
        dtype = arraylike[0].dtype
    if backend == "dask":
        if isinstance(arraylike, da.Array):
            # nothing to be done
            ret_array = arraylike
        elif isinstance(arraylike, np.ndarray):
            ret_array = da.from_array(arraylike.astype(dtype))
        else:
            # use the lambda function to create unique hashes for the dask tasks
            delayed_results = [dask.delayed(lambda x: x)(item) for item in arraylike]
            # delayed_results = [dask.delayed(r) for r in results]
            arrays = [
                da.from_delayed(dr, shape=delayed_shape, dtype=dtype)
                for dr in delayed_results
            ]
            ret_array = da.concatenate(arrays, axis=0)

        if ret_array.dtype != dtype:
            ret_array.astype(dtype)
        if out_shape is not None:
            ret_array = ret_array.reshape(out_shape)

        persist = bool(os.environ.get("DASK_PERSIST_INPUT", False))
        if persist:
            ret_array = ret_array.persist()
        return ret_array
    else:
        if isinstance(arraylike, np.ndarray):
            # nothing to be done
            ret_array = arraylike
        else:
            ret_array = np.array(arraylike)
        if out_shape is not None:
            ret_array = ret_array.reshape(out_shape)
        print(
            f"ret_array.dtype={ret_array.dtype}, dtype={dtype}, {type(ret_array.dtype)}"
        )
        return ret_array.astype(dtype)


@task
def get_files(path: str, fpattern: Optional[list[str]] = ["*.tif"]) -> list[str]:
    """Find files in a directory matching a customizable file name pattern

    Parameters
        ----------
        path: str
            A specific filename or directory to search for files.
        fpattern: str, optional
            Defines a filter for filenames to be included in returned results. Ignored if path refers to a single file. Default; "*.tif".

    Returns
        ----------
        Sorted list of file paths.
    """
    logger = get_run_logger()
    files = []  # ["DUMMY.nd2"]
    logger.info(f"Checking {path}...")
    if os.path.isfile(path):
        files = [path]
    elif os.path.isdir(path):
        for pattern in fpattern:
            files.extend(glob.glob(os.path.join(path, pattern)))
        # remove possible duplicates and sort
        files = list(set(files))
        files.sort()
    logger.info(f"Created list with {len(files)} file(s).")
    return files


@task
def save_movie(
    images: Union[list[np.ndarray], np.ndarray],
    fname: str,
    codec: Optional[str] = "mp4v",
    framerate: Optional[int] = 15,
) -> bool:
    """Saves multi-dimensional numpy array as movie file.

    Parameters
        ----------
        images: list[np.ndarray] or np.ndarray
           The image stack to convert to movie. The assumption is that T is represented in the first axis.
        fname: str
           Filename or full path to save tbe movie to.
        codec: str, optional
           The video codec to use. Default: "mp4v".
        framerate: int, optional
           Sets frames/s for the created video.

    Returns
        ----------
        success: bool
           Returns True if the video creation was successful.
    """
    height, width, _ = images[0].shape
    fourcc = cv2.VideoWriter_fourcc(*codec)
    vout = cv2.VideoWriter()
    success = vout.open(fname, fourcc, framerate, (width, height), True)
    for img in images:
        vout.write(img)
    vout.release()
    return success


def nd2_opener(fname, dask: bool = True) -> tuple[np.array, dict[str, Any]]:
    """Opens a Nikon nd2 file and returns an np.ndarray or dask array plus the file's metadata.

    Parameters
        ----------
        fname: str
            The name of the file to open; can be a full path.
        dask: bool, optional
            If True, returns a dask array. If false returns a np.ndarray. Default: True.

    Returns
        ----------
        imagestack: Union[np.ndarray, dask.array]
            The multi-dimensional image stack.
        metadata; dict[str, Any]
            Relevant metadata like axis shortlabels, pixel_res, pixel_unit, scale.

    """
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
    if dask:
        chunks = (1000, *imagestack.shape[1:])
        print(f"imagestack.shape={imagestack.shape}, chunks={chunks}")
        return da.from_array(imagestack, chunks=chunks), metadata
    else:
        print(f"imagestack.shape={imagestack.shape}")
        return imagestack, metadata


def tif_opener(
    fname: str, dask: Optional[bool] = True
) -> tuple[np.array, dict[str, Any]]:
    """Opens a Tif file and returns an np.ndarray or dask array plus the file's metadata.

    Parameters
        ----------
        fname: str
            The name of the file to open; can be a full path.
        dask: bool, optional
            If True, returns a dask array. If false returns a np.ndarray. Default: True.

    Returns
        ----------
        imagestack: Union[np.ndarray, dask.array]
            The multi-dimensional image stack.
        metadata; dict[str, Any]
            Relevant metadata like axis shortlabels, pixel_res, pixel_unit, scale.
    """
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


def bioio_opener(
    fname: str,
    dask: Optional[bool] = True,
    chunk_dims: Optional[tuple[str, ...]] = None,
    reader: Optional[str] = None,  # bioio_ome_tiff.Reader,
) -> tuple[da.Array, dict]:
    """Opens a images with the BioImageIO pckage and returns an np.ndarray or dask array plus the file's metadata.

    Parameters
        ----------
        fname: str
            The name of the file to open; can be a full path.
        dask: bool, optional
            If True, returns a dask array. If false returns a np.ndarray. Default: True.
        chunk_dims: tuple[int, ...], optional
            Specifies the chunks for the dask array. If None, use readers defaults. Only applies if dask==True. Default: None.

    Returns
        ----------
        imagestack: Union[np.ndarray, dask.array]
            The multi-dimensional image stack.
        metadata; dict[str, Any]
            Relevant metadata like axis shortlabels, pixel_res, pixel_unit, scale.
    """
    if dask:
        # not all readers allow to set chunk dimensions
        try:
            img = BioImage(fname, chunk_dims=chunk_dims, reader=reader)
        except:
            img = BioImage(fname, reader=reader)
        imagestack = img.dask_data
        # imagestack = da.from_array(img.data)  # , chunks=chunk_dims)
    else:
        img = BioImage(fname, reader=reader)
        imagestack = img.data
    metadata = {}
    metadata["axes"] = [
        label.upper() for label in img.dims.order
    ]  # "t", "c", "y", "x"]
    metadata["channel_names"] = img.channel_names
    metadata["channel_axis"] = metadata["axes"].index("C")
    metadata["shape"] = img.shape
    metadata["dims"] = img.dims
    metadata["pixel_unit"] = "um"
    metadata["pixel_res"] = (
        img.physical_pixel_sizes.X + img.physical_pixel_sizes.Y
    ) / 2
    metadata["scale"] = f"{1/metadata['pixel_res']} pixels per {metadata['pixel_unit']}"
    return imagestack, metadata


def skip_opener(fname: str, **kwargs) -> tuple[np.array, dict]:
    """Returns a dummy image reader. The parameters are provided to be compatible with related image reader functions. They are ignored.

    Parameters
        ----------
        fname: str
            The name of the file to open; can be a full path. Ignored.
        dask: bool, optional
            If True, returns a dask array. If false returns a np.ndarray. Default: True. Ignored.

    Returns
        ----------
        result: tuple
            Always returns (None, None).
    """
    return (None, None)


def get_opener(fname) -> Callable[..., Any]:
    """Guesses and returns an image read function based on extension of the fname parameter.

    Parameters
        ----------
        fname: str
            The name of the file to open; can be a full path.

    Returns
        ----------
        func: function
            The image reader function.
    """
    return bioio_opener
    ext = os.path.splitext(fname)[1]
    if ext == ".nd2":
        return nd2_opener
    elif ext in [".tif", ".tiff"]:
        return tif_opener
    return skip_opener


@task
def read_file(
    fname: str,
    dask: Optional[bool] = False,
    squeeze: Optional[bool] = False,
) -> tuple[np.array, dict[str, Any]]:
    """Opens image files and returns an np.ndarray or dask array plus the file's metadata. Reads environment
    variable DASK_CHUNK_DIMS to set chunks for dask arrays; uses bioio_base.dimensions.DEFAULT_CHUNK_DIMS as default.
    For small image files it can be beneficial to persist the dask array representation of the image data in distributed memory.
    This can be set by setting the environment variable DASK_PERSIST_INPUT=True.

    Parameters
        ----------
        fname: str
            The name of the file to open; can be a full path.
        dask: bool, optional
            If True, returns a dask array. If false returns a np.ndarray. Default: True.
        squeeze: bool, optional
            If True, removes single-dimensional axis. E.g. An array shaped as (200, 2, 1, 512, 512) will become shaped (200, 2, 512, 512). Default: False.

    Returns
        ----------
        imagestack: Union[np.ndarray, dask.array]
            The multi-dimensional image stack.
        metadata; dict[str, Any]
            Relevant metadata like axis shortlabels, pixel_res, pixel_unit, scale.

    """
    logger = get_run_logger()

    persist = bool(os.environ.get("DASK_PERSIST_INPUT", False))
    chunk_dims = os.environ.get("DASK_CHUNK_DIMS", DEFAULT_CHUNK_DIMS)
    if chunk_dims and isinstance(chunk_dims, str):
        chunk_dims = list(
            chunk_dims
        )  # should be list, some readers cannot deal with tuples

    opener = get_opener(fname)
    imagestack, metadata = opener(fname, dask=dask, chunk_dims=chunk_dims)

    if squeeze:
        remove_idx = [i for i, dim in enumerate(imagestack.shape) if dim == 1]
        imagestack = np.squeeze(imagestack)
        metadata["shape"] = imagestack.shape
        metadata["axes"] = [
            axis for i, axis in enumerate(metadata["axes"]) if i not in remove_idx
        ]
        metadata["channel_axis"] = metadata["axes"].index("C")
        logger.info(
            f"imagestack.shape={imagestack.shape}, remove_idx={remove_idx}, metadata={metadata}"
        )

    if dask:
        current_chunks = imagestack.chunksize
        new_chunks = [
            n if metadata["axes"][i] in chunk_dims else 1
            for i, n in enumerate(imagestack.shape)
        ]
        if current_chunks != new_chunks:
            # rechunk
            imagestack = imagestack.rechunk(new_chunks)
            logger.info(
                f"original imagestack.chunksize={current_chunks}, applying chunk_dims={chunk_dims} -> new imagestack.chunksize={imagestack.chunksize}"
            )
        if persist:
            logger.info("Persisting image data.")
            imagestack = imagestack.persist()

    logger.info(metadata)
    return imagestack, metadata
