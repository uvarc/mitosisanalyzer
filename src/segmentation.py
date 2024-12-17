import cv2
import numpy as np
from scipy.ndimage import label
from cellpose import models, io

# from cellpose.io import imread


def crop_image(
    img,
    crop_origin=None,
    crop_width=None,
    crop_height=None,
    x_index=1,
    y_index=0,
    copy=False,
):
    """crops numpy array; crop_origin (x,y) tuple"""
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
        crop_width = img_width
    if crop_y + crop_height > img_height:
        crop_height = img_height
    return (
        img[crop_y : crop_y + crop_height, crop_x : crop_x + crop_width],
        (
            crop_x,
            crop_y,
        ),
        crop_width,
        crop_height,
    )


def separate_masks(combined_mask, erode=0, exclude=[255]):
    masks = []
    # print (f'ncc={ncc}, unique={np.unique(lbl)}')
    print(f"Unique values in combined mask: {np.unique(combined_mask)}")
    for i in np.unique(combined_mask):
        if i == 0 or i in exclude:
            continue
        mask = np.zeros(combined_mask.shape, np.uint8)
        mask[combined_mask == i] = 255
        if erode > 0:
            mask = cv2.erode(mask, None, iterations=erode)
        print(f"mask.dtype={mask.dtype}")
        masks.append(mask)
    return masks


def watershed(a, img, dilate=5, erode=0, relthr=0.7):
    """Separate joined objects in binary image via watershed"""
    border = cv2.dilate(img, None, iterations=dilate)
    border = border - cv2.erode(border, None)

    dt = cv2.distanceTransform(img, cv2.DIST_L2, 3)
    dt = ((dt - dt.min()) / (dt.max() - dt.min()) * 255).astype(np.uint8)
    _, dt = cv2.threshold(dt, relthr * 255, 255, cv2.THRESH_BINARY)
    lbl, ncc = label(dt)
    lbl = lbl * (255 / (ncc + 1))
    # Completing the markers now.
    lbl[border == 255] = 255

    lbl = lbl.astype(np.int32)
    odt = lbl.astype(np.uint8)
    markers = cv2.watershed(a, lbl)

    lbl[lbl == -1] = 0
    lbl = lbl.astype(np.uint8)
    masks = separate_masks(lbl, erode=erode)
    return odt, 255 - lbl, masks


def find_embryos(
    channelstack,
    channel=0,
    threshold=0.0,
    cellpose=False,
    cellpose_diam=None,
    model="cyto3",
    erode=0,
):
    print(f"channelstack.shape={channelstack.shape}")
    med = np.median(channelstack, axis=0)
    if cellpose:
        io.logger_setup()
        print(
            f"Using Cellpose {model} model (diameter={cellpose_diam}) to find embryo contours"
        )
        model = models.Cellpose(gpu=True, model_type=model)
        channels = [[0, 0]]
        print(np.array([med, med]).shape)
        raw_masks, flows, styles, diams = model.eval(
            [np.array([med, med])],
            diameter=cellpose_diam,  # 157
            channels=channels,
            cellprob_threshold=1.0,
        )
        print(f"len(raw_masks)={len(raw_masks)}")
        # raw_masks are float, scale to 0-254 (exclude 255 reserved for borders) and convert to uint8
        raw_masks = [(m * 254).astype(np.uint8) for m in raw_masks]
        masks = separate_masks(raw_masks[0], erode=erode)
        # dt, embryo, masks = watershed(
        #    cv2.merge([masks[0], masks[0], masks[0]]), masks[0], relthr=0.8
        # )
    else:
        med = cv2.normalize(med, None, 0, 255.0, norm_type=cv2.NORM_MINMAX)
        med = med.astype(np.uint8)
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        # kernel = np.array([[-1, -1, -1],[-1, 8, -1],[-1, -1, -1]])
        f2d = cv2.filter2D(med, -1, kernel)
        # cv2.imwrite(f"f2d-embryo.tif", f2d)

        if threshold == 0.0:
            print("Using Otsu thresholding")
            __, th = cv2.threshold(
                f2d, 0.75 * 255, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
        else:
            print(f"Using relative threshold of {threshold}")
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
        dt, embryo, masks = watershed(
            cv2.merge([th, th, th]), th, relthr=0.8, erode=erode
        )
        # cv2.imshow('embryo overlay', cv2.merge([embryo,med,embryo]))
        # cv2.waitKey(0)
    for i, mask in enumerate(masks):
        print(
            f"masks[{i}].shape: {masks[i].shape}, dtype={masks[i].dtype}, max={masks[i].max()}, mean={masks[i].mean()}"
        )
        # cv2.imwrite(
        #    f"mask-{i}-cellpose-{cellpose}.tif",
        #    mask,
        # )
    return masks


def rotate_image(image, angle, center=None):
    """Rotate image around image reference point"""
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    image = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    if center is not None:
        dx = (image.shape[1] / 2) - center[0]
        dy = (image.shape[0] / 2) - center[1]
        # print (center, dx,dy, angle)

        transl_mat = np.float32([[1, 0, dx], [0, 1, dy]])
        image = cv2.warpAffine(image, transl_mat, (image.shape[1], image.shape[0]))
    else:
        center = (image.shape[1] / 2, image.shape[0] / 2)
    return image
