"""Shared helpers. The crop format here MUST match DigitSegmenter.kt:
32x32 grayscale, ink = 255 (white) on background = 0 (black), the digit scaled to fit
24x24 (4 px padding) keeping its aspect ratio, centred, resampled with a box (area) filter."""

import numpy as np
from PIL import Image

IMG = 32
PAD = 4
NUM_CLASSES = 11  # 0-9 are digits, 10 means "not a digit"
JUNK = 10
CLASS_NAMES = [str(i) for i in range(10)] + ["junk"]


def normalize_crop(ink: Image.Image) -> Image.Image | None:
    """ink: grayscale image, white ink on black. Returns a 32x32 'L' image, or None if empty."""
    box = ink.getbbox()
    if box is None:
        return None
    ink = ink.crop(box)
    w, h = ink.size
    scale = (IMG - 2 * PAD) / max(w, h)
    nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
    ink = ink.resize((nw, nh), Image.Resampling.BOX)
    canvas = Image.new("L", (IMG, IMG), 0)
    canvas.paste(ink, ((IMG - nw) // 2, (IMG - nh) // 2))
    return canvas


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def make_interpreter(model_path=None, model_content=None):
    """TFLite interpreter for checking the exported model on your computer.
    Uses the newer ai_edge_litert package if installed (pip install ai-edge-litert), else TensorFlow's."""
    try:
        from ai_edge_litert.interpreter import Interpreter
    except ImportError:
        import tensorflow as tf

        Interpreter = tf.lite.Interpreter
    return Interpreter(model_path=model_path, model_content=model_content)
