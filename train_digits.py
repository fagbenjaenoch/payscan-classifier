#!/usr/bin/env python3
"""Train the handwritten-digit classifier and export nuban_digits.tflite.

Expected folder layout (one folder per writer, one sub-folder per digit):
    data/ada/0/001.png ... data/ada/9/050.png        32x32 PNG, white ink on black
    data/ada/junk/001.png                             optional: real "not a digit" crops

    python train_digits.py --data data --out out
    python train_digits.py --data data --out out --mnist      # optional pre-training on MNIST
"""

import argparse
import json
from pathlib import Path

import keras
import numpy as np
import tensorflow as tf
from keras import layers
from PIL import Image, ImageDraw, ImageFilter

from digit_utils import (
    CLASS_NAMES,
    IMG,
    JUNK,
    NUM_CLASSES,
    make_interpreter,
    normalize_crop,
    softmax,
)


def load_dataset(root: Path):
    xs, ys, ws = [], [], []
    for writer_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for label_dir in sorted(p for p in writer_dir.iterdir() if p.is_dir()):
            name = label_dir.name.lower()
            if name.isdigit() and len(name) == 1:
                label = int(name)
            elif name in ("junk", "10"):
                label = JUNK
            else:
                continue
            for f in sorted(label_dir.glob("*.png")):
                img = Image.open(f).convert("L")
                if img.size != (IMG, IMG):
                    img = img.resize((IMG, IMG), Image.Resampling.BOX)
                xs.append(np.asarray(img, dtype=np.uint8))
                ys.append(label)
                ws.append(writer_dir.name)
    if not xs:
        raise SystemExit(
            f"No images found under {root}. Expected {root}/<writer>/<digit>/*.png"
        )
    return np.stack(xs)[..., None], np.array(ys), np.array(ws)


def split_by_writer(writers, val_frac, test_frac, seed):
    """Whole writers go to validation or test, never both train and test."""
    names = sorted(set(writers))
    rng = np.random.default_rng(seed)
    if len(names) < 5:
        print(
            f"\n!! Only {len(names)} writers: falling back to a RANDOM split. Results will look better "
            "than they really are. Collect at least 5 (ideally 30+) writers.\n"
        )
        idx = rng.permutation(len(writers))
        n_test, n_val = int(len(idx) * test_frac), int(len(idx) * val_frac)
        test, val = np.zeros(len(writers), bool), np.zeros(len(writers), bool)
        test[idx[:n_test]] = True
        val[idx[n_test : n_test + n_val]] = True
        return ~(test | val), val, test
    order = list(rng.permutation(names))
    n_test = max(1, round(test_frac * len(names)))
    n_val = max(1, round(val_frac * len(names)))
    test_w, val_w = set(order[:n_test]), set(order[n_test : n_test + n_val])
    print(
        f"Writers: {len(names) - n_test - n_val} train, {n_val} validation {sorted(val_w)}, "
        f"{n_test} test {sorted(test_w)}"
    )
    w = np.array(writers)
    test = np.isin(w, list(test_w))
    val = np.isin(w, list(val_w))
    return ~(test | val), val, test


def make_junk(n, rng, digit_pool):
    """Synthetic 'not a digit' crops: specks, ruled-line pieces, arcs, top/bottom halves of digits, blanks."""
    out = []
    for _ in range(n):
        img = Image.new("L", (IMG, IMG), 0)
        d = ImageDraw.Draw(img)
        kind = rng.integers(0, 6)
        if kind == 0:  # specks
            for _ in range(rng.integers(1, 5)):
                x, y, r = rng.integers(3, 29), rng.integers(3, 29), rng.integers(1, 3)
                d.ellipse([x - r, y - r, x + r, y + r], fill=255)
        elif kind == 1:  # piece of a ruled line
            y, x0, x1 = rng.integers(6, 26), rng.integers(0, 10), rng.integers(22, 32)
            d.line(
                [(x0, y), (x1, y + rng.integers(-2, 3))],
                fill=255,
                width=int(rng.integers(1, 3)),
            )
        elif kind == 2:  # arc
            x0, y0 = rng.integers(0, 10), rng.integers(0, 10)
            d.arc(
                [x0, y0, x0 + rng.integers(12, 24), y0 + rng.integers(12, 24)],
                rng.integers(0, 180),
                rng.integers(180, 360),
                fill=255,
                width=int(rng.integers(1, 3)),
            )
        elif kind == 3 and len(digit_pool):  # top or bottom 40% of a real digit
            src = digit_pool[rng.integers(0, len(digit_pool))][..., 0].copy()
            cut = int(IMG * 0.4)
            if rng.random() < 0.5:
                src[cut:] = 0
            else:
                src[: IMG - cut] = 0
            img = Image.fromarray(src)
        elif kind == 4:  # tiny scribble
            pts = [
                (int(rng.integers(8, 24)), int(rng.integers(8, 24))) for _ in range(4)
            ]
            d.line(pts, fill=255, width=1)
        # kind 5: blank
        if rng.random() < 0.5:
            img = img.filter(ImageFilter.GaussianBlur(0.7))
        a = np.asarray(img, dtype=np.float32)
        a = np.clip(a + rng.normal(0, 6, a.shape) * (rng.random() < 0.5), 0, 255)
        out.append(a.astype(np.uint8)[..., None])
    return np.stack(out)


def load_mnist(limit=30000):
    try:
        (x, y), _ = keras.datasets.mnist.load_data()
    except Exception as e:  # no internet, etc.
        print("MNIST not available, skipping pre-training:", e)
        return None, None
    xs, ys = [], []
    for img, lab in zip(x[:limit], y[:limit]):
        c = normalize_crop(Image.fromarray(img))
        if c is not None:
            xs.append(np.asarray(c, dtype=np.uint8))
            ys.append(int(lab))
    return np.stack(xs)[..., None], np.array(ys)


class RandomThickness(layers.Layer):
    """Training only: randomly thicken or thin the strokes (whole batch at once)."""

    def call(self, x, training=False):
        if not training:
            return x
        r = tf.random.uniform([])
        dil = tf.nn.max_pool2d(x, 3, 1, "SAME")
        ero = -tf.nn.max_pool2d(-x, 3, 1, "SAME")
        return tf.where(r < 0.25, dil, tf.where(r > 0.75, ero, x))

    def compute_output_shape(self, s):
        return s


def build_core():
    inp = keras.Input((IMG, IMG, 1))
    x = inp
    for filters in (32, 64, 128):
        x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.MaxPooling2D()(x)
    x = layers.Flatten()(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(NUM_CLASSES)(
        x
    )  # raw scores ("logits"); softmax is added at export
    return keras.Model(inp, out, name="core")


def build_train_model(core):
    aug = keras.Sequential(
        [
            layers.RandomRotation(0.06, fill_mode="constant", fill_value=0.0),
            layers.RandomTranslation(0.08, 0.08, fill_mode="constant", fill_value=0.0),
            layers.RandomZoom(0.12, fill_mode="constant", fill_value=0.0),
            RandomThickness(),
            layers.GaussianNoise(0.03),
        ],
        name="augment",
    )
    model = keras.Sequential([keras.Input((IMG, IMG, 1)), aug, core])
    model.compile(
        # optimizer=keras.optimizers.Adam(1e-3),  # ignore
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    return model


def fit(model, x, y, xv, yv, epochs, lr=None):
    if lr:
        model.optimizer.learning_rate = lr
    cbs = [
        keras.callbacks.EarlyStopping(
            patience=6, restore_best_weights=True, monitor="val_loss"
        ),
        keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3),
    ]
    model.fit(
        x,
        y,
        validation_data=(xv, yv),
        epochs=epochs,
        batch_size=64,
        callbacks=cbs,
        verbose=2,
    )


# ----------------------------------------------------------------------------- evaluation
def fit_temperature(logits, y):
    """Find T that makes softmax(logits / T) honest about its confidence (lowest log-loss)."""
    best_t, best = 1.0, 1e9
    for t in np.linspace(0.5, 5.0, 46):
        p = softmax(logits / t)
        nll = -np.log(p[np.arange(len(y)), y] + 1e-9).mean()
        if nll < best:
            best, best_t = nll, float(t)
    return best_t


def report(name, probs, y):
    pred = probs.argmax(1)
    acc = (pred == y).mean()
    digit_mask = y < 10
    digit_acc = (
        (pred[digit_mask] == y[digit_mask]).mean() if digit_mask.any() else float("nan")
    )
    print(f"\n=== {name}: {len(y)} images ===")
    print(f"accuracy (all classes): {acc:.4f}    digit accuracy: {digit_acc:.4f}")
    print(
        f"=> estimated chance that a whole 10-digit number is read right: {digit_acc**10:.2%} "
        f"(before checksum repair and multi-frame averaging)"
    )
    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), int)
    for t, p in zip(y, pred):
        cm[t, p] += 1
    print("\nconfusion matrix (rows = true, columns = predicted):")
    print("      " + " ".join(f"{c:>5}" for c in CLASS_NAMES))
    for i in range(NUM_CLASSES):
        if cm[i].sum():
            print(f"{CLASS_NAMES[i]:>5} " + " ".join(f"{v:>5}" for v in cm[i]))
    pairs = sorted(
        (
            (cm[i, j], i, j)
            for i in range(NUM_CLASSES)
            for j in range(NUM_CLASSES)
            if i != j and cm[i, j]
        ),
        reverse=True,
    )[:5]
    if pairs:
        print(
            "\nmost common mistakes: "
            + ", ".join(f"{CLASS_NAMES[i]}->{CLASS_NAMES[j]} x{n}" for n, i, j in pairs)
        )
    conf = probs.max(1)
    print("\nconfidence      images   accuracy")
    for lo, hi in [(0, 0.45), (0.45, 0.6), (0.6, 0.8), (0.8, 0.95), (0.95, 1.01)]:
        m = (conf >= lo) & (conf < hi)
        if m.any():
            print(
                f"{lo:.2f}-{min(hi, 1):.2f}   {m.sum():>8}   {(pred[m] == y[m]).mean():.3f}"
            )
    return digit_acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="out")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument(
        "--val-frac",
        type=float,
        default=0.15,
        help="fraction of WRITERS for validation",
    )
    ap.add_argument(
        "--test-frac",
        type=float,
        default=0.15,
        help="fraction of WRITERS for the final test",
    )
    ap.add_argument(
        "--junk",
        type=int,
        default=1500,
        help="synthetic 'not a digit' images for training",
    )
    ap.add_argument(
        "--mnist", action="store_true", help="pre-train on MNIST first (needs internet)"
    )
    ap.add_argument(
        "--quantize",
        action="store_true",
        help="shrink the .tflite (dynamic-range quantization)",
    )
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    keras.utils.set_random_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    x, y, w = load_dataset(Path(a.data))
    print(f"Loaded {len(y)} images from {len(set(w))} writers")
    tr, va, te = split_by_writer(w, a.val_frac, a.test_frac, a.seed)
    xtr, ytr, xva, yva, xte, yte = x[tr], y[tr], x[va], y[va], x[te], y[te]

    pool = xtr[ytr < 10]
    for n, (xs, ys) in (
        (a.junk, (xtr, ytr)),
        (a.junk // 5, (xva, yva)),
        (a.junk // 5, (xte, yte)),
    ):
        j = make_junk(n, rng, pool)
        xs_new = np.concatenate([xs, j])
        ys_new = np.concatenate([ys, np.full(len(j), JUNK)])
        if xs is xtr:
            xtr, ytr = xs_new, ys_new
        elif xs is xva:
            xva, yva = xs_new, ys_new
        else:
            xte, yte = xs_new, ys_new
    print(f"train {len(ytr)}  validation {len(yva)}  test {len(yte)}")
    f = lambda arr: arr.astype("float32") / 255.0

    core = build_core()
    model = build_train_model(core)
    print(f"model parameters: {core.count_params():,}")

    if a.mnist:
        mx, my = load_mnist()
        if mx is not None:
            print("Pre-training on MNIST...")
            fit(model, f(mx), my, f(xva[yva < 10]), yva[yva < 10], epochs=5)
    print("Training on your data...")
    fit(model, f(xtr), ytr, f(xva), yva, a.epochs, lr=3e-4 if a.mnist else None)

    val_logits = core.predict(f(xva), verbose="0")
    T = fit_temperature(val_logits, yva)
    print(
        f"\ncalibration temperature T = {T:.2f} (confidences are divided by this before softmax)"
    )
    report("VALIDATION", softmax(val_logits / T), yva)
    digit_acc = report(
        "TEST (writers the model never saw)",
        softmax(core.predict(f(xte), verbose="0") / T),
        yte,
    )

    # ---- export: core + temperature + softmax -> one .tflite taking 0..1 floats, giving 11 probabilities
    export = keras.Sequential(
        [keras.Input((IMG, IMG, 1)), core, layers.Rescaling(1.0 / T), layers.Softmax()]
    )
    converter = tf.lite.TFLiteConverter.from_keras_model(export)
    if a.quantize:
        converter.optimizations = {tf.lite.Optimize.DEFAULT}
    tflite = converter.convert()
    assert isinstance(tflite, bytes)
    (out / "nuban_digits.tflite").write_bytes(tflite)
    export.save(out / "nuban_digits.keras")
    np.savez_compressed(out / "test_set.npz", x=xte, y=yte)
    (out / "nuban_digits.meta.json").write_text(
        json.dumps(
            {
                "input": [1, IMG, IMG, 1],
                "input_range": "0..1 float, ink = 1.0, background = 0.0",
                "classes": CLASS_NAMES,
                "temperature": T,
                "test_digit_accuracy": float(digit_acc),
                "train_writers": int(len(set(w[tr]))),
                "train_images": int(len(ytr)),
                "quantized": a.quantize,
            },
            indent=2,
        )
    )
    print(f"\nWrote {out / 'nuban_digits.tflite'} ({len(tflite) / 1024:.0f} KB)")

    # ---- self-check: the .tflite must agree with the Keras model
    interp = make_interpreter(model_content=tflite)
    interp.allocate_tensors()
    i_in, i_out = (
        interp.get_input_details()[0]["index"],
        interp.get_output_details()[0]["index"],
    )
    xs = f(xte[:300])
    ref = export.predict(xs, verbose="0")
    got = []
    for img in xs:
        interp.set_tensor(i_in, img[None])
        interp.invoke()
        got.append(interp.get_tensor(i_out)[0])
    got = np.array(got)
    print(
        f"tflite vs keras: max probability difference {np.abs(ref - got).max():.5f}, "
        f"same top class on {(ref.argmax(1) == got.argmax(1)).mean():.1%} of images"
    )


if __name__ == "__main__":
    main()
