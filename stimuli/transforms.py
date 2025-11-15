import io
import numpy as np
from PIL import Image as PILImage, ImageDraw
from kivy.core.image import Image as CoreImage
from rng_utils import _rng
from config import IMAGE_TILE


def apply_rotation(arr: np.ndarray, angle: float) -> np.ndarray:
    pil = PILImage.fromarray(arr)
    rotated = pil.rotate(angle, expand=False, fillcolor=(255, 255, 255))
    return np.array(rotated)


def apply_occlusion(arr: np.ndarray, occlusion_fraction: float = 0.25, seed: int = None) -> np.ndarray:
    pil = PILImage.fromarray(arr).convert("RGB")
    draw = ImageDraw.Draw(pil)
    w, h = pil.size
    rng = _rng(seed)
    ow = int(w * occlusion_fraction * rng.uniform(0.8, 1.2))
    oh = int(h * occlusion_fraction * rng.uniform(0.8, 1.2))
    x0 = int(rng.uniform(0, max(1, w - ow)))
    y0 = int(rng.uniform(0, max(1, h - oh)))
    draw.rectangle([x0, y0, x0 + ow, y0 + oh], fill=(255, 255, 255))
    return np.array(pil)


def numpy_to_texture(arr: np.ndarray):
    """Convert HxW or HxWx3 uint8 array to a Kivy texture (resized)."""
    if arr.ndim == 2:
        pil = PILImage.fromarray(arr).convert("RGB")
    else:
        pil = PILImage.fromarray(arr, mode="RGB")
    pil = pil.resize(IMAGE_TILE, PILImage.BILINEAR)
    data = io.BytesIO()
    pil.save(data, format="PNG")
    data.seek(0)
    core = CoreImage(data, ext="png")
    return core.texture
