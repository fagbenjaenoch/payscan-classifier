#!/usr/bin/env python3
"""Convert your own pictures of SINGLE digits into the training format:
32x32 grayscale PNG, white digit on black, digit scaled to fit 24x24 without stretching and centred.

    python convert_digits.py --input raw --output data

Folder layouts it understands (the digit label comes from a folder name 0-9, or "junk"):
    raw/<writer>/<digit>/*.jpg        one folder per person, then per digit   (best: enables writer-held-out testing)
    raw/<digit>/*.jpg                 no writer info: everything is filed under --writer (default "imported")
    raw/*.jpg with --label-from-name  file names start with the digit, e.g. 7_001.jpg or 7-12.png

What it does to every picture
  1. fixes phone-photo rotation, flattens transparency, converts colour (blue ink included) to brightness
  2. works out whether the digit is dark-on-light or light-on-dark and flips it to the training polarity
  3. finds the strokes with the SAME adaptive threshold as the app's DigitSegmenter, so shadows and
     uneven lighting do not matter
  4. keeps the digit (plus nearby pieces such as a detached bar) and drops specks and neighbouring shapes
  5. crops to the digit, scales it to fit 24x24 keeping its proportions, centres it in 32x32

Pictures it is not sure about are NOT written to the output folder. They go to <output>_review/ so
you can look at them and decide. A contact sheet and a CSV log go to <output>_report/.
Your input files are never modified.
"""

import argparse
import csv
import random
import re
from collections import Counter
from pathlib import Path
from typing import cast

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageOps
from scipy import ndimage

from digit_utils import IMG, normalize_crop

EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}
MAX_SIDE = 400  # larger pictures are shrunk first: faster, and strokes look like the app's strips
LABELS = set("0123456789") | {"junk"}


# ---------------------------------------------------------------- loading and polarity
def load_gray(path: Path) -> np.ndarray:
    img = ImageOps.exif_transpose(Image.open(path))
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        img = img.convert("RGBA")
        paper = Image.new("RGBA", img.size, (255, 255, 255, 255))
        paper.alpha_composite(img)
        img = paper
    img = img.convert("L")
    if max(img.size) > MAX_SIDE:
        s = MAX_SIDE / max(img.size)
        img = img.resize(
            (max(1, round(img.width * s)), max(1, round(img.height * s))),
            Image.Resampling.BOX,
        )
    return np.asarray(img, dtype=np.uint8)


def otsu(gray: np.ndarray) -> int:
    hist = np.bincount(gray.ravel(), minlength=256).astype(np.float64)
    total, total_sum = gray.size, (np.arange(256) * hist).sum()
    w0 = s0 = 0.0
    best, thr = -1.0, 127
    for i in range(256):
        w0 += hist[i]
        s0 += i * hist[i]
        w1 = total - w0
        if w0 == 0 or w1 == 0:
            continue
        var = w0 * w1 * (s0 / w0 - (total_sum - s0) / w1) ** 2
        if var > best:
            best, thr = var, i
    return thr


def make_ink_dark(gray: np.ndarray):
    """Returns (gray with DARK ink on light paper, note or None).
    Normally the ink is the minority of the pixels. When that is unclear (a bold digit filling the
    picture), the border of the picture decides: the border is background."""
    t = otsu(gray)
    dark_frac = float((gray <= t).mean())
    border = np.concatenate([gray[0], gray[-1], gray[:, 0], gray[:, -1]])
    border_says_ink_dark = float((border <= t).mean()) < 0.5
    minority_says_ink_dark = dark_frac < 0.5
    if dark_frac < 0.35:
        ink_dark = True
    elif dark_frac > 0.65:
        ink_dark = False
    else:
        ink_dark = border_says_ink_dark
    note = (
        "polarity uncertain"
        if (
            0.35 <= dark_frac <= 0.65 and border_says_ink_dark != minority_says_ink_dark
        )
        else None
    )
    return (gray if ink_dark else 255 - gray), note


# ---------------------------------------------------------------- finding the strokes
def ink_mask(gray: np.ndarray) -> np.ndarray:
    """Same rule as DigitSegmenter.binarize: a pixel is ink if it is clearly darker than the average
    of its neighbourhood (window about 1/4 of the picture) AND at least 14 levels darker."""
    h, w = gray.shape
    half = max(7, min(h, w) // 8)
    integral = np.pad(gray.astype(np.int64).cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    ys, xs = np.arange(h), np.arange(w)
    y0, y1 = np.clip(ys - half, 0, h - 1), np.clip(ys + half, 0, h - 1)
    x0, x1 = np.clip(xs - half, 0, w - 1), np.clip(xs + half, 0, w - 1)
    total = (
        integral[np.ix_(y1 + 1, x1 + 1)]
        - integral[np.ix_(y0, x1 + 1)]
        - integral[np.ix_(y1 + 1, x0)]
        + integral[np.ix_(y0, x0)]
    )
    area = (y1 - y0 + 1)[:, None] * (x1 - x0 + 1)[None, :]
    mean = total // area
    return (mean - gray.astype(np.int64)) >= np.maximum(14, mean * 15 // 100)


def keep_digit(mask: np.ndarray):
    """Keep the biggest shape plus pieces close to it. Returns (mask, note or None)."""
    labels, n = cast(
        tuple[NDArray[np.int32], int],
        ndimage.label(mask, structure=np.ones((3, 3))),
    )
    if n == 0:
        return None, "no ink found"
    areas = np.bincount(labels.ravel())[1:]
    main = int(areas.argmax()) + 1
    boxes = ndimage.find_objects(labels)

    def box(i):
        s = boxes[i - 1]
        return s[1].start, s[0].start, s[1].stop - 1, s[0].stop - 1  # x0, y0, x1, y1

    mx0, my0, mx1, my1 = box(main)
    size = max(my1 - my0 + 1, mx1 - mx0 + 1)
    keep, ignored_big = [main], False
    for i in range(1, n + 1):
        if i == main:
            continue
        a = areas[i - 1]
        x0, y0, x1, y1 = box(i)
        gap = max(0, x0 - mx1 - 1, mx0 - x1 - 1, y0 - my1 - 1, my0 - y1 - 1)
        if a >= max(4, 0.08 * areas[main - 1]) and gap <= 0.12 * size:
            keep.append(i)
        elif a >= 0.3 * areas[main - 1]:
            ignored_big = True
    return np.isin(labels, keep), (
        "another large shape was ignored" if ignored_big else None
    )


def convert_one(path: Path):
    """Returns (32x32 PIL image or None, status 'ok'/'review', note)."""
    gray = load_gray(path)
    if gray.shape[0] < 8 or gray.shape[1] < 8:
        return None, "review", "picture too small"
    dark_ink, notes = make_ink_dark(gray)
    notes = [notes] if notes else []
    mask = ink_mask(dark_ink)
    frac = float(mask.mean())
    if frac > 0.55 or frac < 0.01:
        notes.append(f"ink covers {frac:.0%} of the picture")
    mask, note = keep_digit(mask)
    if mask is None:
        return None, "review", "no ink found"
    if note:
        notes.append(note)
    ys, xs = np.nonzero(mask)
    if max(ys.max() - ys.min(), xs.max() - xs.min()) + 1 < 8:
        notes.append("digit is tiny")
    crop = normalize_crop(Image.fromarray((mask * 255).astype(np.uint8)))
    return crop, ("review" if notes else "ok"), "; ".join(notes)


def find_jobs(root: Path, default_writer: str, label_from_name: bool):
    jobs, skipped = [], []
    for p in sorted(root.rglob("*")):
        if not p.is_file() or p.suffix.lower() not in EXTS:
            continue
        parts = p.relative_to(root).parts
        label = writer = None
        if len(parts) >= 3 and parts[-2].lower() in LABELS:
            label, writer = parts[-2].lower(), parts[-3]
        elif len(parts) == 2 and parts[0].lower() in LABELS:
            label, writer = parts[0].lower(), default_writer
        elif label_from_name and re.match(r"^\d(?!\d)", p.name):
            label = p.name[0]
            writer = parts[-2] if len(parts) >= 2 else default_writer
        if label is None:
            skipped.append(p)
        else:
            jobs.append((re.sub(r"[^A-Za-z0-9_-]", "_", writer or ""), label, p))
    return jobs, skipped


def unique(path: Path) -> Path:
    n, cand = 1, path
    while cand.exists():
        cand = path.with_name(f"{path.stem}_{n}{path.suffix}")
        n += 1
    return cand


def contact_sheet(files_by_label, out: Path, per_row=14, scale=3):
    rows = [sorted(files_by_label.get(str(d), [])) for d in range(10)]
    rnd = random.Random(1)
    sheet = Image.new("L", (per_row * IMG * scale, 10 * IMG * scale), 40)
    for r, files in enumerate(rows):
        for c, f in enumerate(rnd.sample(files, min(per_row, len(files)))):
            tile = (
                Image.open(f)
                .convert("L")
                .resize((IMG * scale, IMG * scale), Image.Resampling.NEAREST)
            )
            sheet.paste(tile, (c * IMG * scale, r * IMG * scale))
    sheet.save(out)


def main():
    ap = argparse.ArgumentParser(
        description="Convert pictures of single digits to 32x32 white-on-black training crops."
    )
    ap.add_argument("--input", required=True, help="folder with your pictures")
    ap.add_argument(
        "--output", default="data", help="training folder to write (default: data)"
    )
    ap.add_argument(
        "--writer",
        default="imported",
        help="writer name when the folders have no writer level",
    )
    ap.add_argument(
        "--label-from-name",
        action="store_true",
        help="take the digit from the start of the file name",
    )
    a = ap.parse_args()

    root, out = Path(a.input), Path(a.output)
    review, report = (
        out.with_name(out.name + "_review"),
        out.with_name(out.name + "_report"),
    )
    report.mkdir(parents=True, exist_ok=True)
    jobs, skipped = find_jobs(root, a.writer, a.label_from_name)
    if not jobs:
        raise SystemExit(
            f"No labelled pictures found under {root}. See the layouts at the top of this script."
        )

    log, counts, written = [], Counter(), {}
    for writer, label, path in jobs:
        try:
            crop, status, note = convert_one(path)
        except Exception as e:  # unreadable or corrupt file
            crop, status, note = None, "error", f"{type(e).__name__}: {e}"
        dest = ""
        if crop is not None:
            base = (out if status == "ok" else review) / writer / label
            base.mkdir(parents=True, exist_ok=True)
            target = unique(base / f"{path.stem}.png")
            crop.save(target)
            dest = str(target)
            if status == "ok":
                written.setdefault(label, []).append(target)
        counts[status] += 1
        log.append([str(path), writer, label, status, note, dest])

    with open(report / "conversion_log.csv", "w", newline="") as f:
        csv.writer(f).writerows(
            [["source", "writer", "label", "status", "note", "output"]] + log
        )
    if written:
        contact_sheet(written, report / "contact_sheet.png")

    writers = {w for w, _, _ in jobs}
    print(
        f"\nconverted into {out}/: {counts['ok']}   sent to {review}/ for you to look at: {counts['review']}   "
        f"unreadable: {counts['error']}   skipped (no label found): {len(skipped)}"
    )
    print(
        "images per digit: "
        + "  ".join(f"{d}:{len(written.get(d, []))}" for d in "0123456789")
    )
    print(f"writers: {len(writers)}")
    if len(writers) < 5:
        print(
            "!! Fewer than 5 writers. Training will fall back to a random split and its scores will be too "
            "optimistic. If you know who wrote what, organise the input as raw/<writer>/<digit>/."
        )
    if counts["review"]:
        why = Counter(r[4] for r in log if r[3] == "review")
        print(
            "why pictures went to review: "
            + "; ".join(f"{k or '?'} x{v}" for k, v in why.most_common(4))
        )
    if skipped:
        print(
            f"skipped e.g. {skipped[0]} (not in a folder named 0-9 or junk; try --label-from-name)"
        )
    print(
        f"Open {report / 'contact_sheet.png'}: every digit must be WHITE on a black background and roughly centred."
    )


if __name__ == "__main__":
    main()
