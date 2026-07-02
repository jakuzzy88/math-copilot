"""
render_equation.py – Synthetic handwriting-style renderer

Takes an equation label string and renders it as a greyscale PNG image
with randomised augmentations to simulate handwriting variation:
    - font size variation
    - x/y offset jitter
    - inter-symbol spacing variation
    - slight rotation
    - brightness / contrast adjustment
    - Gaussian blur
    - salt-and-pepper noise

Uses Pillow for rendering. Falls back to the Pillow default font if no
system TrueType fonts are found.

Sprint 2 implementation.
"""

from __future__ import annotations

import math
import os
import random
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

# ---------------------------------------------------------------------------
# Font discovery
# ---------------------------------------------------------------------------

# Common system font directories
_FONT_SEARCH_DIRS = [
    "/usr/share/fonts",
    "/usr/local/share/fonts",
    os.path.expanduser("~/.fonts"),
    os.path.expanduser("~/.local/share/fonts"),
]

# Preferred fonts (roughly handwriting-ish or at least variable)
_PREFERRED_FONTS = [
    "DejaVuSans.ttf",
    "DejaVuSerif.ttf",
    "DejaVuSansMono.ttf",
    "LiberationMono-Regular.ttf",
    "LiberationSans-Regular.ttf",
    "NotoSansMono-Regular.ttf",
    "Ubuntu-Regular.ttf",
    "FreeMono.ttf",
    "FreeSans.ttf",
]


def _discover_fonts() -> list[str]:
    """Walk system font dirs and return paths to usable TrueType fonts."""
    found: list[str] = []
    for font_dir in _FONT_SEARCH_DIRS:
        if not os.path.isdir(font_dir):
            continue
        for root, _dirs, files in os.walk(font_dir):
            for f in files:
                if f.lower().endswith((".ttf", ".otf")):
                    found.append(os.path.join(root, f))
    return found


def _pick_font(font_paths: list[str], rng: random.Random) -> str | None:
    """Pick a font path, preferring the preferred list."""
    if not font_paths:
        return None

    # Try preferred first
    preferred = [p for p in font_paths if any(pf in p for pf in _PREFERRED_FONTS)]
    if preferred:
        return rng.choice(preferred)
    return rng.choice(font_paths)


# Cache discovered fonts at module level
_SYSTEM_FONTS: list[str] | None = None


def _get_system_fonts() -> list[str]:
    global _SYSTEM_FONTS
    if _SYSTEM_FONTS is None:
        _SYSTEM_FONTS = _discover_fonts()
    return _SYSTEM_FONTS


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _load_font(font_path: str | None, size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Load a TrueType font or fall back to the Pillow default."""
    if font_path:
        try:
            return ImageFont.truetype(font_path, size)
        except (OSError, IOError):
            pass
    # Fallback: Pillow default bitmap font (small, but always available)
    return ImageFont.load_default()


def render_equation_image(
    label: str,
    width: int = 512,
    height: int = 128,
    seed: Optional[int] = None,
) -> Image.Image:
    """
    Render an equation label string into a greyscale PIL Image with
    random augmentations.

    Args:
        label: The equation string, e.g. "3x+4=10".
        width: Target image width in pixels.
        height: Target image height in pixels.
        seed: Random seed for reproducible augmentation.

    Returns:
        A Pillow Image in mode 'L' (greyscale).
    """
    rng = random.Random(seed)
    np_rng = np.random.RandomState(seed if seed is not None else rng.randint(0, 2**31))

    # --- augmentation parameters ---
    font_size = rng.randint(max(20, height // 5), max(28, height // 3))
    x_offset = rng.randint(5, max(6, width // 10))
    y_offset = rng.randint(5, max(6, height // 6))
    char_spacing = rng.randint(0, 6)
    rotation_deg = rng.uniform(-3.0, 3.0)
    brightness_factor = rng.uniform(0.7, 1.0)
    blur_radius = rng.choice([0, 0, 0, 0.5, 1.0])
    noise_amount = rng.uniform(0.0, 0.05)

    # --- font ---
    fonts = _get_system_fonts()
    font_path = _pick_font(fonts, rng)
    font = _load_font(font_path, font_size)

    # --- create canvas (white background, dark text) ---
    # Work on a larger canvas for rotation, then crop
    pad = 40
    canvas_w = width + pad * 2
    canvas_h = height + pad * 2
    img = Image.new("L", (canvas_w, canvas_h), 255)
    draw = ImageDraw.Draw(img)

    # --- draw characters with spacing ---
    cursor_x = x_offset + pad
    cursor_y = y_offset + pad
    for ch in label:
        # Small per-character y jitter
        jy = rng.randint(-2, 2)
        draw.text((cursor_x, cursor_y + jy), ch, fill=0, font=font)
        # Measure character width
        try:
            bbox = font.getbbox(ch)
            ch_w = bbox[2] - bbox[0]
        except AttributeError:
            ch_w = font_size // 2  # fallback for default font
        cursor_x += ch_w + char_spacing

    # --- rotation ---
    if abs(rotation_deg) > 0.1:
        img = img.rotate(
            rotation_deg,
            resample=Image.BICUBIC,
            expand=False,
            fillcolor=255,
        )

    # --- crop back to target size ---
    left = pad
    top = pad
    img = img.crop((left, top, left + width, top + height))

    # --- brightness / contrast ---
    arr = np.array(img, dtype=np.float32)
    # Brightness: scale towards white (255)
    arr = arr * brightness_factor + 255 * (1 - brightness_factor)
    arr = np.clip(arr, 0, 255)

    # --- blur ---
    if blur_radius > 0:
        img = Image.fromarray(arr.astype(np.uint8), "L")
        img = img.filter(ImageFilter.GaussianBlur(radius=blur_radius))
        arr = np.array(img, dtype=np.float32)

    # --- noise ---
    if noise_amount > 0:
        noise = np_rng.normal(0, noise_amount * 255, arr.shape)
        arr = arr + noise
        arr = np.clip(arr, 0, 255)

    return Image.fromarray(arr.astype(np.uint8), "L")


def render_and_save(
    label: str,
    image_path: str | Path,
    label_path: str | Path,
    width: int = 512,
    height: int = 128,
    seed: Optional[int] = None,
) -> None:
    """
    Render an equation and save the image + label to disk.

    Args:
        label: Equation string.
        image_path: Path to save the .png image.
        label_path: Path to save the .txt label file.
        width: Image width.
        height: Image height.
        seed: Random seed.
    """
    image_path = Path(image_path)
    label_path = Path(label_path)

    image_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.parent.mkdir(parents=True, exist_ok=True)

    img = render_equation_image(label, width=width, height=height, seed=seed)
    img.save(str(image_path), "PNG")

    with open(label_path, "w", encoding="utf-8") as f:
        f.write(label)


if __name__ == "__main__":
    # Quick test: render a sample equation
    out_dir = Path(__file__).resolve().parent / "_test_output"
    out_dir.mkdir(exist_ok=True)
    render_and_save(
        "3x+4=10",
        out_dir / "sample.png",
        out_dir / "sample.txt",
        seed=42,
    )
    print(f"Rendered sample → {out_dir / 'sample.png'}")
