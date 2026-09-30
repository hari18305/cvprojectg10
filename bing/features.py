"""Normed-gradient (NG) feature extraction.

The BING feature of a window is the 8x8 map of gradient magnitudes obtained by
resizing the window to 8x8 pixels.  In practice the *whole image* is resized
once per quantised window size (w, h) so that every 8x8 patch of the resized
NG map corresponds to one w x h window of the original image.
"""
import cv2
import numpy as np

FEAT = 8  # side of the NG feature (8x8 = 64-D)


def to_colorspace(img_bgr, space="RGB"):
    """Convert a BGR uint8 image to the colour space used for gradients."""
    if space in ("RGB", "BGR"):
        return img_bgr
    if space == "HSV":
        return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    if space == "LAB":
        return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    if space == "GRAY":
        return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    raise ValueError(f"unknown colour space {space}")


def normed_gradient(img, kernel="simple"):
    """Normed gradient map g = min(|gx| + |gy|, 255) as uint8.

    For multi-channel images the channel with the largest gradient is used
    (as in the original BING implementation).

    kernel: "simple" -> 1-D [-1, 0, 1] mask (the paper's choice)
            "sobel"  -> 3x3 Sobel operator
    """
    img = img.astype(np.int16) if img.dtype == np.uint8 else img
    if kernel == "simple":
        gx = np.zeros(img.shape, np.int16)
        gy = np.zeros(img.shape, np.int16)
        gx[:, 1:-1] = img[:, 2:] - img[:, :-2]
        gx[:, 0] = (img[:, 1] - img[:, 0]) * 2
        gx[:, -1] = (img[:, -1] - img[:, -2]) * 2
        gy[1:-1] = img[2:] - img[:-2]
        gy[0] = (img[1] - img[0]) * 2
        gy[-1] = (img[-1] - img[-2]) * 2
        gx, gy = np.abs(gx), np.abs(gy)
    elif kernel == "sobel":
        f = img.astype(np.float32)
        gx = np.abs(cv2.Sobel(f, cv2.CV_32F, 1, 0, ksize=3)) / 4.0
        gy = np.abs(cv2.Sobel(f, cv2.CV_32F, 0, 1, ksize=3)) / 4.0
    else:
        raise ValueError(kernel)
    if gx.ndim == 3:  # max over colour channels
        gx = gx.max(axis=2)
        gy = gy.max(axis=2)
    return np.minimum(gx.astype(np.int32) + gy, 255).astype(np.uint8)


def resize(img, w, h, interp="area"):
    """Resize; "area" (anti-aliased) when shrinking, or "linear" everywhere
    (faster, as in the original C++ implementation)."""
    if interp == "area" and w < img.shape[1] and h < img.shape[0]:
        flag = cv2.INTER_AREA
    else:
        flag = cv2.INTER_LINEAR
    return cv2.resize(img, (int(w), int(h)), interpolation=flag)


def box_feature(img, box, kernel="simple", feat=FEAT, interp="area"):
    """64-D NG feature of a single box (x1, y1, x2, y2) in a colour-converted image.

    The box is expanded by one feature-pixel on each side before resizing so that
    the gradient at the 8x8 border sees real neighbours, exactly as happens when
    the whole image is resized at test time.
    """
    H, W = img.shape[:2]
    x1, y1, x2, y2 = [float(v) for v in box]
    bw, bh = max(x2 - x1, 2.0), max(y2 - y1, 2.0)
    px, py = bw / feat, bh / feat
    X1, Y1 = int(round(x1 - px)), int(round(y1 - py))
    X2, Y2 = int(round(x2 + px)), int(round(y2 + py))
    pad = [max(0, -Y1), max(0, Y2 - H), max(0, -X1), max(0, X2 - W)]
    crop = img[max(Y1, 0):min(Y2, H), max(X1, 0):min(X2, W)]
    if any(pad):
        crop = cv2.copyMakeBorder(crop, *pad, cv2.BORDER_REPLICATE)
    g = normed_gradient(resize(crop, feat + 2, feat + 2, interp), kernel)
    return g[1:-1, 1:-1].reshape(-1).astype(np.float32)
