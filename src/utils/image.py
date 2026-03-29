import base64
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter


def resize_for_vlm(image, max_dim=1024):
    if isinstance(image, str):
        image = Image.open(image)
    w, h = image.size
    if max(w, h) <= max_dim:
        return image
    scale = max_dim / max(w, h)
    return image.resize((int(w * scale), int(h * scale)), Image.LANCZOS)


def enhance_contrast(image, factor=1.5):
    if isinstance(image, str):
        image = Image.open(image)
    return ImageEnhance.Contrast(image).enhance(factor)


def binarize_sauvola(image, window_size=25, k=0.2):
    """Sauvola adaptive thresholding for historical document preprocessing.
    Useful for faded ink and uneven illumination."""
    if isinstance(image, str):
        image = Image.open(image)
    gray = np.array(image.convert("L"), dtype=np.float64)
    h, w = gray.shape

    # integral image for fast local mean/variance
    integral = np.cumsum(np.cumsum(gray, axis=0), axis=1)
    integral_sq = np.cumsum(np.cumsum(gray ** 2, axis=0), axis=1)

    pad = window_size // 2
    output = np.zeros_like(gray, dtype=np.uint8)

    for i in range(h):
        for j in range(w):
            y1, x1 = max(0, i - pad), max(0, j - pad)
            y2, x2 = min(h - 1, i + pad), min(w - 1, j + pad)
            area = (y2 - y1 + 1) * (x2 - x1 + 1)

            s = integral[y2, x2]
            sq = integral_sq[y2, x2]
            if y1 > 0:
                s -= integral[y1 - 1, x2]
                sq -= integral_sq[y1 - 1, x2]
            if x1 > 0:
                s -= integral[y2, x1 - 1]
                sq -= integral_sq[y2, x1 - 1]
            if y1 > 0 and x1 > 0:
                s += integral[y1 - 1, x1 - 1]
                sq += integral_sq[y1 - 1, x1 - 1]

            mean = s / area
            var = max(0, sq / area - mean ** 2)
            std = np.sqrt(var)
            threshold = mean * (1 + k * (std / 128 - 1))
            output[i, j] = 255 if gray[i, j] > threshold else 0

    return Image.fromarray(output)


def preprocess_page(image_path, enhance=True, binarize=False):
    """Standard preprocessing pipeline for a scanned page."""
    img = Image.open(image_path).convert("RGB")
    if enhance:
        img = ImageEnhance.Sharpness(img).enhance(1.3)
        img = enhance_contrast(img, 1.3)
    if binarize:
        img = binarize_sauvola(img)
    return img


def encode_image_base64(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def crop_region(image, bbox):
    if isinstance(image, str):
        image = Image.open(image)
    return image.crop(bbox)


def horizontal_projection(image):
    """Compute horizontal projection profile for line segmentation fallback."""
    if isinstance(image, str):
        image = Image.open(image)
    gray = np.array(image.convert("L"))
    binary = (gray < 128).astype(np.int32)
    return binary.sum(axis=1)


def detect_lines_projection(image, min_gap=10, min_height=15):
    """Detect text line boundaries using horizontal projection profile.
    Returns list of (y_start, y_end) tuples."""
    profile = horizontal_projection(image)
    threshold = profile.max() * 0.05

    in_line = False
    lines = []
    start = 0

    for y, val in enumerate(profile):
        if val > threshold and not in_line:
            start = y
            in_line = True
        elif val <= threshold and in_line:
            if y - start >= min_height:
                lines.append((start, y))
            in_line = False

    if in_line and len(profile) - start >= min_height:
        lines.append((start, len(profile)))

    # merge lines that are too close (probably split by noise)
    merged = []
    for line in lines:
        if merged and line[0] - merged[-1][1] < min_gap:
            merged[-1] = (merged[-1][0], line[1])
        else:
            merged.append(line)

    return merged
