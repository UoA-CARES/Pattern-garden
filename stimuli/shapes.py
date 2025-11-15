import numpy as np
from PIL import Image as PILImage, ImageDraw
from rng_utils import _rng


def draw_shape(shape: str, size: int = 192, seed: int = None, variant: dict = None) -> np.ndarray:
    rng = _rng(seed)
    variant = variant or {}
    bg = (255, 255, 255)
    img = PILImage.new("RGB", (size, size), bg)
    d = ImageDraw.Draw(img)

    padding = int(size * variant.get("padding_frac", 0.15))
    color = tuple(int(x) for x in rng.randint(50, 200, size=3))
    w = int(variant.get("stroke", 6))

    if shape == "square":
        x0, y0 = padding, padding
        x1, y1 = size - padding, size - padding
        d.rectangle([x0, y0, x1, y1], outline=color, width=w)

    elif shape == "rectangle":
        d.rectangle([padding, size//3, size - padding, 2*size//3], outline=color, width=w)

    elif shape == "diamond":
        cx, cy = size//2, size//2
        pts = [(cx, padding), (size - padding, cy), (cx, size - padding), (padding, cy)]
        d.polygon(pts, outline=color)
        d.line(pts + [pts[0]], fill=color, width=w)

    elif shape == "circle":
        d.ellipse([padding, padding, size - padding, size - padding], outline=color, width=w)

    elif shape == "oval":
        d.ellipse([padding, size//4, size - padding, 3*size//4], outline=color, width=w)

    elif shape == "triangle":
        p1 = (size // 2, padding)
        p2 = (padding, size - padding)
        p3 = (size - padding, size - padding)
        d.polygon([p1, p2, p3], outline=color)
        d.line([p1, p2, p3, p1], fill=color, width=w)

    elif shape == "pentagon":
        cx, cy, r = size//2, size//2, size//2 - padding
        pts = [(cx + r*np.cos(2*np.pi*i/5 - np.pi/2),
                cy + r*np.sin(2*np.pi*i/5 - np.pi/2)) for i in range(5)]
        d.polygon(pts, outline=color)
        d.line(pts + [pts[0]], fill=color, width=w)

    elif shape == "hexagon":
        cx, cy, r = size//2, size//2, size//2 - padding
        pts = [(cx + r*np.cos(2*np.pi*i/6 - np.pi/2),
                cy + r*np.sin(2*np.pi*i/6 - np.pi/2)) for i in range(6)]
        d.polygon(pts, outline=color)
        d.line(pts + [pts[0]], fill=color, width=w)

    elif shape == "star":
        cx, cy, r = size // 2, size // 2, size // 2 - padding
        pts = []
        for i in range(10):
            ang = np.radians(36 * i - 90)
            rr = r if i % 2 == 0 else r * 0.45
            pts.append((cx + rr * np.cos(ang), cy + rr * np.sin(ang)))
        d.polygon(pts, outline=color)
        d.line(pts + [pts[0]], fill=color, width=w)

    elif shape == "cross":
        thickness = size // 6
        d.rectangle([size//2 - thickness//2, padding,
                     size//2 + thickness//2, size - padding], outline=color, width=w)
        d.rectangle([padding, size//2 - thickness//2,
                     size - padding, size//2 + thickness//2], outline=color, width=w)

    elif shape == "arrow":
        # simple upward arrow
        d.line([size//2, padding, size//2, size - padding], fill=color, width=w)
        d.polygon([(size//2, padding),
                   (size//2 - size//6, padding + size//6),
                   (size//2 + size//6, padding + size//6)], outline=color, fill=color)


    else:
        # fallback = square
        d.rectangle([padding, padding, size - padding, size - padding], outline=color, width=w)

    # Optional occlusion
    if variant.get("occlude", False):
        frac = float(variant.get("oc_frac", 0.2))
        ow, oh = int(size * frac), int(size * frac)
        x0 = int(rng.uniform(0, max(1, size - ow)))
        y0 = int(rng.uniform(0, max(1, size - oh)))
        d.rectangle([x0, y0, x0 + ow, y0 + oh], fill=bg)

    # Optional rotation
    rot = int(variant.get("rotate", 0))
    if rot:
        img = img.rotate(rot, resample=PILImage.BICUBIC, expand=False, fillcolor=bg)

    return np.array(img)
