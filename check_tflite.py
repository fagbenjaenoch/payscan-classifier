#!/usr/bin/env python3
"""Run the exported model on PNG crops and print what it thinks.

python check_tflite.py out/nuban_digits.tflite crop1.png crop2.png ...
python check_tflite.py out/nuban_digits.tflite --folder some_dir_of_pngs --expect-from-name
"""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from digit_utils import CLASS_NAMES, IMG, make_interpreter

ap = argparse.ArgumentParser()
ap.add_argument("model")
ap.add_argument("files", nargs="*")
ap.add_argument("--folder")
a = ap.parse_args()
files = [Path(f) for f in a.files] + (
    sorted(Path(a.folder).glob("*.png")) if a.folder else []
)

interp = make_interpreter(model_path=a.model)
interp.allocate_tensors()
i_in, i_out = (
    interp.get_input_details()[0]["index"],
    interp.get_output_details()[0]["index"],
)
for f in files:
    img = Image.open(f).convert("L").resize((IMG, IMG), Image.Resampling.BOX)
    x = (np.asarray(img, dtype=np.float32) / 255.0)[None, ..., None]
    interp.set_tensor(i_in, x)
    interp.invoke()
    p = interp.get_tensor(i_out)[0]
    top = p.argsort()[::-1][:2]
    print(
        f"{f.name}: {CLASS_NAMES[top[0]]} ({p[top[0]]:.2f})   runner-up {CLASS_NAMES[top[1]]} ({p[top[1]]:.2f})"
    )
