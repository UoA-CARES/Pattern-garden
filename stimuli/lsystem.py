import numpy as np
from PIL import Image as PILImage, ImageDraw, ImageFilter, ImageOps
from rng_utils import _rng


def lsystem_image(size=192, seed=None, iterations=4):
    rng = _rng(seed)
    axiom = "F"
    turn = int(rng.choice([60, 75, 90, 120]))

    def rule(ch: str) -> str:
        return {"F": "F[+F]F[-F]F"}.get(ch, ch)

    seq = axiom
    for _ in range(iterations):
        seq = ''.join(rule(ch) for ch in seq)

    img = PILImage.new("L", (size, size), 255)
    draw = ImageDraw.Draw(img)
    x, y = size // 2, size // 2
    angle = -90
    step = max(2, size // 40)
    stack = []
    for ch in seq:
        if ch == "F":
            nx = x + step * np.cos(np.radians(angle))
            ny = y + step * np.sin(np.radians(angle))
            draw.line((x, y, nx, ny), fill=0, width=1)
            x, y = nx, ny
        elif ch == "+":
            angle += turn
        elif ch == "-":
            angle -= turn
        elif ch == "[":
            stack.append((x, y, angle))
        elif ch == "]" and stack:
            x, y, angle = stack.pop()
    img = img.filter(ImageFilter.GaussianBlur(radius=0.5))
    img = ImageOps.autocontrast(img, cutoff=2)
    return np.array(img.convert("RGB"))
