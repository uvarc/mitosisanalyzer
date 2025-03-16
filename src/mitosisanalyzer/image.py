import cv2
import joblib
import math
import numpy as np
import dask
import dask.array as da
import time
import skimage as ski
from scipy.ndimage import label
from cellpose import models, io
from prefect import task, runtime
from prefect.logging import get_run_logger
from typing import Optional, Union, Literal
from .utils import create_array


@task
def adjust_intensity(
    img: np.ndarray,
    axes: Optional[Union[str, tuple[str], tuple[int]]] = None,
    axes_labels: Optional[Union[str, tuple[str]]] = None,
    mode: Optional[Literal["stretch", "adaptive"]] = "stretch",
    clip_limit: Optional[float] = 0.025,
    dtype=np.uint8,
    verbose: Optional[int] = 100,
) -> np.ndarray:
    """Adjusts image intensity levels based on image input's histogram.

    Parameters
        ----------
        img: np.ndarray
            Array representing the input image. Can be multi-dimensional.
        axes: str | tuple[str] | tuple[int], optional
            Specifies the axes indices (i.e. img.shape) to process by. For example: Assuming TCYX stack, axes=(-2, -1) would iterate over the outer TC axes
            to process 512x512 YX planes defined in the last two axes.The same could be expressed axes=(2,3). If the axes_labels are provided, the same
            could be expressed with axes=("Y,"X") or axes="YX". If None, adjust based on intensity values of entire stack. Default: None
        axes_labels: str | tuple[str], optional
            Defines the axes labels of the img array. For example, "TCYX" or ("T", "C", "Y', "X").
        mode: str, optional
            Specifies the mode for adjusting the image. Options: "stretch', "adaptive"
        clip_limit: float, optional
            Relative intensity values (0.0-0.5) that will be clipped at the histogram's low and high intensity tail ends.
        verbose; int, optional
            Level of verbosity of joblib.Parallel task progress. Default: 0, no messages

    Returns
        ----------
        adj_img: np.ndarray
            A copy of the input image with the adjusted contrast. The dtype is np.uint8
    """
    backend = runtime.flow_run.parameters["backend"]  # for parallelizing with joblib
    logger = get_run_logger()
    logger.info(
        f"Adjusting image, backend={backend}, type(img)={type(img)}, img.dtype={img.dtype}, img.shape={img.shape}, axes={axes}, axes_labels={axes_labels}, mode={mode}"
    )

    low = int(255 * clip_limit)
    high = int(255 * (1.0 - clip_limit))
    if axes is not None:
        # process based on plane-by-plane histograms
        if isinstance(axes, str):
            axes = list(axes)
        if isinstance(axes_labels, str):
            axes_labels = list(axes_labels)
        if all(isinstance(item, str) for item in axes):
            if axes is None:
                raise ValueError(
                    f"Axes are given as {axes} but axes_labels are missing to determine axes order."
                )
            flat_shape = [img.shape[i] for i, a in enumerate(axes_labels) if a in axes]
            flat_shape.insert(0, -1)
        else:
            flat_shape = [img.shape[i] for i in axes]
            flat_shape.insert(0, -1)
        logger.info(f"flat_shape={flat_shape}")
        orig_shape = img.shape
        flat_img = img.reshape(flat_shape)
        logger.info(f"flat_img.shape={flat_img.shape}, type(flat_img)={type(flat_img)}")
        axis = 0
        if mode == "stretch":
            with joblib.parallel_config(backend=backend):  # , prefer="threads"):
                results = joblib.Parallel(verbose=verbose)(
                    joblib.delayed(ski.exposure.rescale_intensity)(
                        i, out_range=(low, high)
                    )
                    for i in flat_img
                )
            adj_img = create_array(
                results,
                dtype=dtype,
                delayed_shape=flat_shape[1:],
                out_shape=orig_shape,
                backend=backend,
            )
            # adj_img = adj_img.rechunk(img.chunksize)
        elif mode == "adaptive":
            with joblib.parallel_config(backend=backend):
                results = np.array(
                    joblib.Parallel(verbose=verbose)(
                        joblib.delayed(ski.exposure.equalize_adapthist)(
                            i, clip_limit=clip_limit
                        )
                        for i in flat_img
                    )
                )
            adj_img = create_array(
                results,
                dtype=dtype,
                delayed_shape=flat_shape[1:],
                out_shape=orig_shape,
                backend=backend,
            )
        else:
            raise NotImplementedError(
                "The {mode} mode is not implemented. available options: stretch, adaptive"
            )
    else:
        # process based on whole stack histogram
        if mode == "stretch":
            adj_img = ski.exposure.rescale_intensity(img, in_range=(low, high))
        elif mode == "adaptive":
            adj_img = ski.exposure.equalize_adapthist(img, clip_limit=clip_limit)
        else:
            raise NotImplementedError(
                "The {mode} mode is not implemented. available options: stretch, adaptive"
            )
    # adj_img is float64, convert to uint8
    # adj_img = ski.util.img_as_ubyte(adj_img)
    logger.info(
        f"type(adj_img)={type(adj_img)}, adj_img.dtype={adj_img.dtype}, adj_img.shape={adj_img.shape}"
    )
    return adj_img


def get_contour(
    binary: np.ndarray, min_area: Optional[int] = None, max_area: Optional[int] = None
) -> Union[np.ndarray, None]:
    """Determines the contour of the lagest object in a binary mask.

    Parameters
        ----------
        binary: np.ndarray
            A 2-d array representing the binary image.
        min_area: int, optional
            The minimal area of the object to be considered.
        max_area: int, optional
            The maximal area of the object to be considered.

    Returns
        ----------
        contour: np.ndarray | None
            The array defining the contour or None if the no contour was found with min_area <= area <= max-area.
    """
    contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    for i, contour in enumerate(contours):
        # Calculate the area of each contour
        area = cv2.contourArea(contour)
        # Ignore contours that are too small or too large
        if (min_area is None or area >= min_area) and (
            max_area is None or area <= max_area
        ):
            return contour
    return None


@task
def slice_stack(
    stack: Union[np.ndarray, da.Array],
    axes: list[str],
    slice_at: dict[str, int],
    strict: Optional[bool] = True,
) -> Union[np.ndarray, da.Array]:
    """Slices a numpy or dask array along a given axis.

    Paramters
        ----------
        stack: np.ndarray | da.Array
            The array to be sliced.
        axes: list[str]
            The list of axes short labels, e.g. ["t", "c", "z", "y", "x"]
        slice_at: dict[str, int]
            Defines the axis label and slice boundaries, e.g. {"c": 0} extracts the first channel from the array.
        strict: bool, optional
            If False, ignores keys in slice_at that do not exist in axes short labels. Default: False.

    Returns
        ----------
        result: np.ndarray | da.Array
            The sliced array (numpy aor dask, depending on the type of the input stack)
    """
    logger = get_run_logger()

    if not strict:
        # remove k:v pairs that do not exist in axes short labels
        n_slice_at = {k: v for k, v in slice_at.items() if k in axes}
    else:
        n_slice_at = slice_at
    if len(n_slice_at) != 1 or list(n_slice_at.keys())[0] not in axes:
        raise ValueError(
            f"Expect single key but {len(slice_at)} keys provided for slice_at {slice_at}, or key  not found in axes labels."
        )

    result = stack[:, list(n_slice_at.values())[0]]
    logger.info(
        f"Input {stack.shape}, slicing {n_slice_at}, returning {type(result)} {result.shape}"
    )
    return result


@task
def register_stack(imagestack: np.ndarray):
    """Performs a registration on images along top level axis. Not implemented yet.

    Parameters
        ----------
        imagestack : np.ndarray
            The arrays should have at least three axes, i.e. representing a 3-dimensional
            or higher dimensional image.
    """
    logger = get_run_logger()
    logger.info(f"Running registration for stack.shape={imagestack.shape}")
    time.sleep(1)
    return imagestack


@task
def to_RGB(
    channelstacks: Union[np.ndarray, list[np.ndarray]],
    axes_labels: Optional[Union[str, tuple[str]]] = None,
    colors: Optional[dict[int, str]] = {"R": 0, "G": 1, "B": 2},
    verbose: Optional[int] = 100,
) -> np.ndarray:
    """Merges a list or array of single channel image stacks into an RGB image stack. Cannot merge more than 3 image stacks.

    Parameters
        ----------
        channelstacks: np.ndarray | list[np.ndarray]
            A list of single channel imagestacks. Each satck can be TYX or ZYX.
        axes_labels: str | tuple[str], optional
            Specify order of axes, e.g. "TCYX" or ("T","C","Y","X").
        colors: dict, optional
            Mapping of "R", "G", "B" channels to indices in channelstacks. Default: {"G": 0, "R": 1}
        verbose; int, optional
            Level of verbosity of joblib.Parallel task progress. Default: 0, no messages

    Returns
        ----------
        rgb: np.ndarray
            The merged image stack.
    """
    backend = runtime.flow_run.parameters["backend"]  # for parallelizing with joblib
    logger = get_run_logger()

    channels = len(channelstacks)
    if axes_labels is not None and "C" in axes_labels and axes_labels.index("C") != 0:
        channel_index = axes_labels.index("C")
        channelstacks = np.array(channelstacks)
        channelstacks = np.moveaxis(channelstacks, channel_index, 0)
        channels = len(channelstacks)
        logger.debug(
            f"channelstacks.shape={channelstacks.shape}, channels (first axis)={channels}"
        )
    if 0 < channels <= 3:
        height = channelstacks[0].shape[-2]  # Y axis
        width = channelstacks[0].shape[-1]  # X axis
        blank = np.zeros(channelstacks[0].shape[1:], np.uint8)
        logger.debug(f"channelstacks[0].shape={channelstacks[0].shape}")
        with joblib.parallel_config(backend=backend):
            rgb = np.array(
                joblib.Parallel(verbose=verbose)(
                    joblib.delayed(cv2.merge)(
                        [
                            (
                                channelstacks[colors["B"]][i].astype(np.uint8)
                                if colors["B"] < channels
                                else blank
                            ),
                            (
                                channelstacks[colors["G"]][i].astype(np.uint8)
                                if colors["G"] < channels
                                else blank
                            ),
                            (
                                channelstacks[colors["R"]][i].astype(np.uint8)
                                if colors["R"] < channels
                                else blank
                            ),
                        ]
                    )
                    for i in range(channelstacks[0].shape[0])
                )
            )
            # [
            #    cv2.merge(
            #        [
            #            (
            #                channelstacks[colors["B"]][i].astype(np.uint8)
            #                if colors["B"] < channels
            #                else blank
            #            ),
            #            (
            #                channelstacks[colors["G"]][i].astype(np.uint8)
            #                if colors["G"] < channels
            #                else blank
            #            ),
            #            (
            #                channelstacks[colors["R"]][i].astype(np.uint8)
            #                if colors["R"] < channels
            #                else blank
            #            ),
            #        ]
            #    )
            #    for i in range(channelstacks[0].shape[0])
            # ]
        # )
    else:
        raise ValueError("Cannot handle more than three channel stacks")
    logger.info(f"rgb.shape={rgb.shape}, rgb.dtype={rgb.dtype}")
    return rgb.squeeze()


def crop_stack(
    imagestack: Union[np.ndarray, list[np.ndarray], da.Array, list[da.Array]],
    width: int,
    height: int,
    x: Optional[int] = 0,
    y: Optional[int] = 0,
    center: Optional[bool] = False,
) -> Union[np.ndarray, da.Array]:
    """Crops the x-y plane of an image stack.

    Parameters
        ----------
        imagestack: np.ndarray | da.Array
            The stack to crop. The array has to be 2-dimensional or higher.
        width: int
            The width of the cropped image stack.
        height: int
            The height of the cropped image stack.
        x: int, optional
            The x coorindate of the top left corner of the cropping box. Ignored if center==True. Default: 0.
        y: int, optional
            The y coorindate of the top left corner of the cropping box. Ignored if center==True. Default: 0.
        center: bool
            If True, centers the cropping box on the image based on the width and height parameters. Overrides the x and y parameters.
            Default: False.

    Returns
        ----------
        cropped: np.ndarray
            The cropped image stack.
    """
    if center:
        x1 = int((imagestack.shape[-2] - width) / 2)
        y1 = int((imagestack.shape[-1] - height) / 2)
    else:
        x1 = x
        y1 = y
    x2 = x1 + width
    y2 = y1 + height
    print(f"x1={x1}, x2={x2}, y1={y1}, y2={y2}")
    cropped = imagestack[..., y1:y2, x1:x2]
    return cropped


def crop_image(
    img: np.ndarray,
    crop_origin: Optional[tuple[int, int]] = None,
    crop_width: Optional[int] = None,
    crop_height: Optional[int] = None,
    x_index: Optional[int] = 1,
    y_index: Optional[int] = 0,
    copy: Optional[bool] = False,
) -> np.ndarray:
    """Crops an image to a specified bounding box.

    Parameters
        ----------
        img: np.ndarray
            Image to be cropped.
        crop_origin: tuple, optional
            Sets the top left corner (x,y) of the cropping box. If value is None, the crop_origin will
            be calculated such that the cropping box will be centered on the image.
        crop_width: int, optional
            Sets the width of the cropping box. If crop_width is None, the image will not be cropped along the x-axis.
        crop_height: int, optional
            Sets the height of the cropping box. If crop_height is None, the image will not be cropped along the y-axis.
        x_index: int, optional
            Defines the index of the x-axis in the array. Default: 1
        y_index: int, optional
            Defines the index of the y-axis in the array. Default: 0

    Returns
        cropped_img: np.ndarray
            The cropped image.
        crop_origin: tuple[int, int]
            The effective crop_origin (x,y).
        crop_width: int
            The effective crop width
        crop_height: int
            The effective crop height
    """
    if copy:
        img = img.copy()
    img_width = img.shape[x_index]
    img_height = img.shape[y_index]
    if crop_origin is None and crop_width is None and crop_height is None:
        return (
            img,
            (
                0,
                0,
            ),
            img_width,
            img_height,
        )
    if crop_width is None or crop_width > img_width:
        crop_width = img_width
    if crop_height is None or crop_height > img_height:
        crop_height = img_height
    if crop_origin is None:
        # crop around center
        crop_x = (img_width - crop_width) // 2
        crop_y = (img_height - crop_height) // 2
    else:
        crop_x, crop_y = crop_origin
    crop_x = max(crop_x, 0)
    crop_y = max(crop_y, 0)
    if crop_x + crop_width > img_width:
        crop_width = img_width - crop_x
    if crop_y + crop_height > img_height:
        crop_height = img_height - crop_y
    return (
        img[crop_y : crop_y + crop_height, crop_x : crop_x + crop_width],
        (
            crop_x,
            crop_y,
        ),
        crop_width,
        crop_height,
    )


def separate_masks(
    combined_mask: np.ndarray,
    erode: Optional[int] = 0,
    exclude: Optional[list[int]] = [255],
) -> list[np.ndarray]:
    """
    Converts an image ith multiple labeled regions into a list of binary masks.
    The assumption is that each region is defined by pixels of the same intensity value.
    The number of unique intensity values corrsponds to the number of regions.

    Parameters
        ----------
        combined_mask: np.ndarray
            Image with region labels. Each region represented by pixels with matching intensity values.
        erode: int, optional
            Applies n iterations of a 1 pixel erosion. Default: 0.
        exclude: list[int], optional
            Regions with pixel values that are found in this list are excluded from the returned list of masks.

    Returns
        ----------
        mask: list[np.ndarray]
            List of binary images. Each binary image represening a mask of a single region.
    """
    masks = []
    for i in np.unique(combined_mask):
        if i == 0 or i in exclude:
            continue
        mask = np.zeros(combined_mask.shape, np.uint8)
        mask[combined_mask == i] = 255
        if erode > 0:
            mask = cv2.erode(mask, None, iterations=erode)
        masks.append(mask)
    return masks


def watershed(
    a,
    img: np.ndarray,
    dilate: Optional[int] = 5,
    erode: Optional[int] = 0,
    relthr: Optional[float] = 0.7,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Separate joined objects in binary image via watershed.

    Parameters
        ----------
        a:
        img: np.ndarray
            Binary image to apply the watershed to.
        dilate: int, optional
            Applies n iterations of a 1 pixel dilation to the input image before watershed processing.
            This defines the thickness of a border for each region. Default: 5.
        erode: int, optional
            Applies n iterations of a 1 pixel erosion after the watershed to shrink all regions. Default: 0.
        relthr: float, optional
            Defines the relative threshold (0<=relthr<=1.0) to be applied to the distance transformation image.
            Pixels with distance transform values greater than the relative threshold will be included in
            the result image. Default: 0.7
    Returns
        ----------
        odt: np.ndarray
            Image with inverted region labels. Each region is defined by pixels with identical pixel values.
        i_lbl: np.ndarray
            Image with inverted region labels. Each region is defined by pixels with identical pixel values.
        masks: list[np.ndarray]
            List of binary images. Each binary image represening a mask of a single region.
    """
    border = cv2.dilate(img, None, iterations=dilate)
    border = border - cv2.erode(border, None)

    dt = cv2.distanceTransform(img, cv2.DIST_L2, 3)
    dt = ((dt - dt.min()) / (dt.max() - dt.min()) * 255).astype(np.uint8)
    _, dt = cv2.threshold(dt, relthr * 255, 255, cv2.THRESH_BINARY)
    lbl, no_regions = label(dt)
    lbl = lbl * (255 / (no_regions + 1))
    # Completing the markers now.
    lbl[border == 255] = 255

    lbl = lbl.astype(np.int32)
    odt = lbl.astype(np.uint8)
    markers = cv2.watershed(a, lbl)

    lbl[lbl == -1] = 0
    lbl = lbl.astype(np.uint8)
    masks = separate_masks(lbl, erode=erode)
    return odt, 255 - lbl, masks


@task
def find_cells(
    imagestack: Union[np.ndarray, da.Array],
    threshold: Optional[float] = 0.0,
    cell_diameter: Optional[float] = None,
    area_tolerance: Optional[tuple[float, float]] = (0.7, 1.3),
    cellpose: Optional[bool] = False,
    model: Optional[str] = "cyto3",
    erode: Optional[int] = 0,
) -> list[np.ndarray]:
    """Finds cells masks in image stack using either Cellpose or conventional segmentation methods.
    The assumption is that indiviudal cells in the imagestack remain aligned along the T or Z axis. The cell segmentation
    mask is determined on a single focal plane image obtained by median projection of the stack.

    Parameters
        ----------
        imagestack: np.ndarray | da.Array
            A multi-dimensional imagestack, assumed to be TCYX or ZCYX.
        threshold: float, optional
            Threshold value to binarize the imagestack. A value of 0.0 results in auto-thresholding with otsu method. Default: 0.0.
        cell_diameter: float, optional
            Expected cell diameter in pixel.
        area_tolerance: tuple[float,float], optional
            Defines area ranges (min,max) that are acceptable relative to the area of a circle of given cell_diameter target.
        cellpose: bool, optional
            If True, use Cellpose to segment the image. Otherwise use conventional segmentation methods. Default: False.
        model: str, optional
            Name of the Cellpose model to use. Ignored if cellpose==False. Default: "cyto3".
        erode: int, optional
            Defines an erosion kernel to be applied to the identified cell masks. This can help to eliminate signal at the
            cell borders in downstream image analysis steps.
    Returns
        masks: list
            A list of np.ndarray, each element representing a single cell.
    """
    logger = get_run_logger()

    if cell_diameter is not None:
        min_area = area_tolerance[0] * np.pi * 0.25 * cell_diameter * cell_diameter
        max_area = area_tolerance[1] * np.pi * 0.25 * cell_diameter * cell_diameter
    else:
        min_area = -math.inf
        max_area = math.inf

    logger.info("Starting cell detection")
    # median along TZ axis, i.e. median of all YX image planes in given channel
    # the assumption is that cells do not move
    # wrap with np.array to trigger compute of dask array
    med = np.median(imagestack, axis=range(0, len(imagestack.shape) - 2))
    # med = da.median(imagestack, axis=range(0, len(imagestack.shape) - 2)).compute()
    logger.info(
        f"type(imagestack)={type(imagestack)}, imagestack.shape={imagestack.shape}, type(med)={type(med)}, med.shape={med.shape}"
    )
    if cellpose:
        # process with Cellpose
        io.logger_setup()
        logger.info(
            f"Using Cellpose {model} model (diameter={cell_diameter}) to find cell contours"
        )
        model = models.Cellpose(gpu=True, model_type=model)
        channels = [[0, 0]]
        raw_masks, flows, styles, diams = model.eval(
            [np.array([med, med])],
            diameter=cell_diameter,  # 157
            channels=channels,
            cellprob_threshold=1.0,
        )
        logger.debug(f"len(raw_masks)={len(raw_masks)}")
        # raw_masks are float, scale to 0-254 (exclude 255 reserved for borders) and convert to uint8
        raw_masks = [(m * 254).astype(np.uint8) for m in raw_masks]
        masks = separate_masks(raw_masks[0], erode=erode)
        # dt, embryo, masks = watershed(
        #    cv2.merge([masks[0], masks[0], masks[0]]), masks[0], relthr=0.8
        # )
    else:
        # process with conventinal methods
        med = cv2.normalize(med, None, 0, 255.0, norm_type=cv2.NORM_MINMAX)
        med = med.astype(np.uint8)
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        # kernel = np.array([[-1, -1, -1],[-1, 8, -1],[-1, -1, -1]])
        f2d = cv2.filter2D(med, -1, kernel)
        # cv2.imwrite(f"f2d-embryo.tif", f2d)

        if threshold == 0.0:
            logger.info("Using Otsu thresholding")
            __, th = cv2.threshold(
                f2d, 0.75 * 255, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
        else:
            logger.info(f"Using relative threshold of {threshold}")
            __, th = cv2.threshold(f2d, threshold * 255, 255, cv2.THRESH_BINARY)
        # cv2.imwrite(f"th-embryo.tif", th)
        # embryo = cv2.adaptiveThreshold(med,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)
        # embryo = cv2.morphologyEx(embryo, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (50, 50)))

        # cv2.imshow('med', med)
        # cv2.imshow('thresholded', th)
        # cv2.imshow('f2d', f2d)

        """
        ret, thresh = cv2.threshold(med,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        # noise removal
        kernel = np.ones((3,3),np.uint8)
        opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)
        # sure background area
        sure_bg = cv2.dilate(opening,kernel,iterations=3)
        # Finding sure foreground area
        dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
        ret, sure_fg = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)
        # Finding unknown region
        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(sure_bg,sure_fg)
        cv2.imshow('sure fg', sure_fg)
        cv2.imshow('uncertain', unknown)
        """

        kernel = np.ones((3, 3), np.uint8)
        # embryo = cv2.morphologyEx(embryo,cv2.MORPH_OPEN, kernel, iterations = 2)
        _, _, masks = watershed(cv2.merge([th, th, th]), th, relthr=0.8, erode=erode)

    for i, mask in enumerate(masks):
        logger.debug(
            f"masks[{i}].shape: {masks[i].shape}, dtype={masks[i].dtype}, max={masks[i].max()}, mean={masks[i].mean()}"
        )
    masks = [m for m in masks if np.sum(m) >= min_area and np.sum(m <= max_area)]
    return masks


def rotate_image(
    image: Union[np.ndarray, da.Array], angle: float, center: tuple[int, int] = None
) -> Union[np.ndarray, da.Array]:
    """Rotate image around image reference point

    Parameters
        ----------
        image: np.ndarray | da.Array
            Array representing the image plane. Expected CYX.
        angle: float
            Rotation angle in radians.
        center: tuple, optional
            Fixed in the image to rotate around.

    Returns
        ----------
        result: np.ndarray | da.Array
            Rotated image array (CYX). Same shape as input image.
    """
    # create matrix to rotate around specified center point
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    # move channel axis to end. This is a requirement by cv2
    image = np.transpose(image, (1, 2, 0))
    rot_image = cv2.warpAffine(
        np.array(image), rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR
    )

    if center is not None:
        # rotation is off-center so the rotated image is translated
        dx = (image.shape[1] / 2) - center[0]
        dy = (image.shape[0] / 2) - center[1]

        transl_mat = np.float32([[1, 0, dx], [0, 1, dy]])
        rot_image = cv2.warpAffine(
            rot_image, transl_mat, (image.shape[1], image.shape[0])
        )
    else:
        center = (image.shape[1] / 2, image.shape[0] / 2)
    result = np.transpose(rot_image, (2, 0, 1))
    return result
