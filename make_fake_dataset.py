#!/usr/bin/env python3
"""Makes a FAKE dataset from computer fonts so you can test the whole pipeline (training,
export, Android) before you have collected any real handwriting.

The accuracy you get on this data says NOTHING about real handwriting.

  python make_fake_dataset.py --out data --writers 12 --per-digit 40
  python make_fake_dataset.py --out data --strips-out fake_strips --strips 30   # also writes test strips
"""

import argparse
import glob
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from digit_utils import normalize_crop


def find_fonts() -> list[str]:
    patterns = [
        "/usr/share/fonts/**/*.ttf",
        "/usr/share/fonts/**/*.otf",
        "/System/Library/Fonts/**/*.ttf",
        "C:/Windows/Fonts/*.ttf",
    ]
    try:
        import matplotlib

        patterns.append(
            str(Path(matplotlib.get_data_path()) / "fonts" / "ttf" / "*.ttf")
        )
    except Exception:
        pass
    fonts = []
    for p in patterns:
        fonts += glob.glob(p, recursive=True)
    # skip symbol / emoji style fonts that have no digits
    bad = ("symbol", "emoji", "dingbat", "cmex", "cmsy", "cmmi", "cmr", "stix")
    return sorted(f for f in set(fonts) if not any(b in f.lower() for b in bad))


def glyph(ch: str, font_path: str | None, rng: random.Random, size=96) -> Image.Image:
    """White glyph on black, randomly slanted, rotated and thickened."""
    font = (
        ImageFont.truetype(font_path, size)
        if font_path
        else ImageFont.load_default(size)
    )
    img = Image.new("L", (size * 2, size * 2), 0)
    ImageDraw.Draw(img).text((size // 2, size // 4), ch, fill=255, font=font)
    slant = rng.uniform(-0.3, 0.3)
    img = img.transform(
        img.size,
        Image.Transform.AFFINE,
        (1, slant, -slant * size, 0, 1, 0),
        Image.Resampling.BILINEAR,
    )
    img = img.rotate(rng.uniform(-8, 8), Image.Resampling.BILINEAR)
    for _ in range(rng.choice([0, 0, 1, 2])):
        img = img.filter(ImageFilter.MaxFilter(3))
    return img


def make_dataset(out: Path, writers: int, per_digit: int, seed: int):
    rng = random.Random(seed)
    fonts = find_fonts()
    if not fonts:
        print("No TTF fonts found; using PIL's default font for every writer.")
        fonts = [None]
    for w in range(writers):
        font = fonts[w % len(fonts)]
        for d in range(10):
            folder = out / f"writer{w:02d}" / str(d)
            folder.mkdir(parents=True, exist_ok=True)
            for i in range(per_digit):
                crop = normalize_crop(glyph(str(d), font, rng))
                if crop is not None:
                    crop.save(folder / f"{i:03d}.png")
    print(f"Wrote {writers} fake writers x 10 digits x {per_digit} images to {out}")


def make_strips(
    out: Path,
    count: int,
    seed: int,
    digits: int = 10,
    height: int = 96,
    font_limit: int = 30,
):
    """Dark ink on light paper, with spacing jitter, touching digits, noise and ruled lines."""
    rng = random.Random(seed)
    fonts = (find_fonts())[:font_limit]
    out.mkdir(parents=True, exist_ok=True)
    for n in range(count):
        text = "".join(rng.choice("0123456789") for _ in range(digits))
        ink = rng.randint(20, 90)
        paper = rng.randint(190, 250)
        font = rng.choice(fonts)
        for _attempt in range(20):  # some fonts have no digit glyphs: try another
            if glyph("8", font, rng).getbbox() is not None:
                break
            font = rng.choice(fonts)
        pieces, x = [], 12
        for ch in text:
            g = glyph(ch, font, rng, size=rng.randint(62, 80))
            box = g.getbbox()
            if box is None:
                continue
            g = g.crop(box)
            scale = rng.uniform(0.9, 1.1) * (height * 0.62) / g.size[1]
            g = g.resize(
                (max(1, round(g.size[0] * scale)), max(1, round(g.size[1] * scale))),
                Image.Resampling.BOX,
            )
            pieces.append(
                (
                    g,
                    x,
                    int(height * 0.2)
                    + rng.randint(-6, 6)
                    + (int(height * 0.62) - g.size[1]) // 2,
                )
            )
            x += g.size[0] + rng.randint(-2, 14)
        width = x + 12
        canvas = Image.new("L", (width, height), paper)
        for g, px, py in pieces:
            mask = g.point(lambda v: int(v * rng.uniform(0.85, 1.0)))
            canvas.paste(Image.new("L", g.size, ink), (px, py), mask)
        if rng.random() < 0.4:  # ruled line near the baseline
            y = int(height * 0.85) + rng.randint(-3, 3)
            ImageDraw.Draw(canvas).line(
                [(0, y), (width, y + rng.randint(-2, 2))], fill=ink + 40, width=2
            )
        arr = np.asarray(canvas, dtype=np.float32) + np.random.default_rng(
            seed + n
        ).normal(0, 4, (height, width))
        img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)).filter(
            ImageFilter.GaussianBlur(0.6)
        )
        img.save(
            out / f"f{fonts.index(font):02d}_strip{n:03d}__{text}.png"
        )  # f<font>_ = the fake "writer"
    print(f"Wrote {count} test strips to {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    ap.add_argument("--writers", type=int, default=12)
    ap.add_argument("--per-digit", type=int, default=40)
    ap.add_argument("--strips-out", default=None)
    ap.add_argument("--strips", type=int, default=30)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    make_dataset(Path(a.out), a.writers, a.per_digit, a.seed)
    if a.strips_out:
        make_strips(Path(a.strips_out), a.strips, a.seed + 1000)
