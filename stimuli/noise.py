import numpy as np
from enum import Enum
from PIL import Image as PILImage, ImageFilter, ImageOps
from rng_utils import _rng


import numpy as np
from PIL import Image as PILImage, ImageFilter
from rng_utils import _rng


def perlin_like_noise_gray(
    size=192,
    octaves=3,
    seed=None,
    freq_start=2,
    persistence=0.5,
    base_blur=1.5,
    median_size=0,
    gamma=1.2,
    difficulty=0.5,  # 0.0 = very easy, 1.0 = very hard
):
    """
    Generate grayscale Perlin-like noise with tunable difficulty.

    Args:
        size (int): Image size.
        octaves (int): Layers of detail. More octaves = more complex patterns.
        seed (int): Random seed.
        freq_start (int): Starting frequency for first octave.
        persistence (float): Decay factor for higher octaves.
        base_blur (float): Gaussian blur for each octave.
        median_size (int): Median filter size (0 to disable).
        gamma (float): Gamma correction (>1 boosts mid-tones).
        difficulty (float): Adaptive difficulty (0 = easier, 1 = harder).

    Returns:
        np.ndarray: Grayscale noise image.
    """
    rng = _rng(seed)
    size = int(size)
    base = np.zeros((size, size), dtype=np.float32)
    freq = max(1, int(freq_start))
    amplitude = 1.0
    total_amp = 0.0

    # Adjust parameters based on difficulty
    octaves = int(np.clip(2 + difficulty * (octaves - 2), 2, octaves))
    persistence = 0.4 + 0.3 * difficulty  # smoother vs rougher
    blur_scale = 1.0 + (1.0 - difficulty)  # easy = more blur

    for _ in range(octaves):
        small_w = max(2, size // freq)
        small_h = max(2, size // freq)
        small = rng.randn(small_h, small_w).astype(np.float32)
        small = (small - small.min()) / (np.ptp(small) + 1e-9)

        noise_img = PILImage.fromarray((small * 255).astype(np.uint8))
        noise_img = noise_img.resize((size, size), resample=PILImage.BICUBIC)

        blur_radius = max(0.5, base_blur / (freq ** 0.5)) * blur_scale
        noise_img = noise_img.filter(ImageFilter.GaussianBlur(radius=blur_radius))

        arr = np.array(noise_img, dtype=np.float32) / 255.0
        arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-9)

        base += arr * amplitude
        total_amp += amplitude

        freq *= 2
        amplitude *= persistence

    base = base / (total_amp + 1e-9)
    base = (base - base.min()) / (base.max() - base.min() + 1e-9)

    # Gamma correction to emphasize mid-contrast
    base = np.power(base, gamma)

    pil_final = PILImage.fromarray((base * 255).astype(np.uint8))

    if median_size and median_size > 1:
        pil_final = pil_final.filter(ImageFilter.MedianFilter(size=median_size))

    return np.array(pil_final, dtype=np.uint8)



def colored_noise(size=192, octaves=3, seed=None, difficulty=0.5,
                  saturation_boost=0.04, desat_blend=0.06, autocontrast_cutoff=1):
    rng = _rng(seed)
    channels = []
    base_seed = rng.randint(0, 1_000_000)
    for ch_idx in range(3):
        ch = perlin_like_noise_gray(
            size=size,
            octaves=octaves,
            seed=int(base_seed + ch_idx * 1013),
            difficulty=difficulty
        )
        channels.append(ch.astype(np.uint8))

    img = np.stack(channels, axis=-1)
    pil = PILImage.fromarray(img, mode="RGB")

    pil = pil.filter(ImageFilter.GaussianBlur(radius=1.0 * (1.0 - 0.5 * difficulty)))
    if autocontrast_cutoff and autocontrast_cutoff > 0:
        pil = ImageOps.autocontrast(pil, cutoff=autocontrast_cutoff)
    pil = PILImage.blend(
        pil,
        PILImage.new("RGB", pil.size, (245, 245, 245)),
        alpha=desat_blend * (1.0 - difficulty)
    )

    try:
        hsv = pil.convert("HSV")
        h_arr = np.array(hsv)
        h_arr[..., 1] = np.clip(
            h_arr[..., 1].astype(np.int16) + int(255 * saturation_boost),
            0, 255
        ).astype(np.uint8)
        pil = PILImage.fromarray(h_arr, mode="HSV").convert("RGB")
    except Exception:
        pass

    return np.array(pil)



def groovy_noise(size=192, seed=None, freq=6.0):
    rng = _rng(seed)
    x = np.linspace(0, 2 * np.pi, size)
    y = np.linspace(0, 2 * np.pi, size)
    xx, yy = np.meshgrid(x, y)

    angle = rng.uniform(0, np.pi)
    pattern = np.sin(freq * (np.cos(angle) * xx + np.sin(angle) * yy))
    pattern = (pattern - pattern.min()) / (pattern.max() - pattern.min())

    arr = (pattern * 255).astype(np.uint8)
    return np.stack([arr, arr, arr], axis=-1)


def alpha_perlin_noise(size=192, seed=None, octaves=2, alpha=0.4, freq=1.2):
    """
    Generate bold, flowing Perlin bands with wide contours.
    """
    base = perlin_like_noise_gray(
        size=size,
        octaves=octaves,
        seed=seed,
        freq_start=int(freq),      # lower frequency = wider bands
        gamma=0.8,                 # stronger contrast
        base_blur=3.0,             # extra smoothing widens shapes
        difficulty=0.3,
    )
    overlay = perlin_like_noise_gray(
        size=size,
        octaves=octaves,
        seed=(seed or 0) + 777,
        freq_start=int(freq * 0.8),
        gamma=0.8,
        base_blur=3.0,
        difficulty=0.3,
    )

    base = base.astype(np.float32) / 255.0
    overlay = overlay.astype(np.float32) / 255.0
    combined = (1 - alpha) * base + alpha * overlay

    # Normalize and convert to thick ridges
    combined = np.sin(combined * np.pi * 1.5)  # fewer oscillations = thicker contours
    combined = np.abs(combined) ** 0.4         # widen bright areas

    combined = (combined - combined.min()) / (combined.max() - combined.min() + 1e-9)
    arr = (combined * 255).astype(np.uint8)
    return np.stack([arr, arr, arr], axis=-1)



class NoiseFamily(Enum):
    COLORED = "colored"
    GROOVY = "groovy"
    ALPHA_PERLIN = "alpha_perlin"

    def generate(self, size=192, seed=None, difficulty=0.5):
        """
        Generate a noise image from the selected family.
        Args:
            size (int): Image size in pixels.
            seed (int): Random seed for reproducibility.
            difficulty (float): 0.0 (very easy) → 1.0 (very hard).
        Returns:
            np.ndarray: Noise image (H×W×3).
        """
        if self is NoiseFamily.COLORED:
            return colored_noise(size=size, seed=seed, octaves=2, difficulty=difficulty)
        elif self is NoiseFamily.GROOVY:
            return groovy_noise(size=size, seed=seed, freq=4.0 + 6.0 * difficulty)
        elif self is NoiseFamily.ALPHA_PERLIN:
            return alpha_perlin_noise(
                size=size,
                seed=seed,
                octaves=2,
                alpha=0.4 + 0.2 * difficulty,
                freq=1.0 + 1.5 * (1.0 - difficulty)   # easier = broader
            )
        else:
            raise ValueError(f"Unknown family {self}")
