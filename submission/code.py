#!/usr/bin/env python3
"""
Final inference script for the dual-token complex FFT model.

Input:  .npy file of shape (T, 6, 3, 120, 2)
Output: .jsonl file, one record per frame:
        {"frame": t, "localizations": [[x, y], ...]}
"""

import argparse
import importlib.util
import io
import json
import re
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf


ROOM_X_MAX = 4.8
ROOM_Y_MAX = 7.2
N_SLOTS = 4
TFLITE_MODEL_METADATA_NAME = b"EEAI_MODEL_PARAMS"

FFT_BIN_LO = 5
FFT_BIN_HI = 48
FFT_FREQ_LO = 0
FFT_FREQ_HI = 25
SUB_WINDOW_SIZE = 50
BACKGROUND_MODE = "ema"
EMA_ALPHA = 0.96


def parse_args():
    parser = argparse.ArgumentParser(
        description="Inference for the final dual-token complex FFT TFLite model"
    )
    parser.add_argument("--input-path", required=True, help="Input .npy file")
    parser.add_argument("--output-path", required=True, help="Output .jsonl file")
    parser.add_argument(
        "--model-dir",
        default=None,
        help="Directory containing model.tflite (default: this script's directory)",
    )
    parser.add_argument(
        "--model-path",
        default=None,
        help="Path to a .tflite model; overrides --model-dir/model.tflite",
    )
    return parser.parse_args()


def apply_ema_iq(iq_data, alpha):
    if iq_data.shape[0] == 0:
        return np.empty_like(iq_data)

    bg = iq_data[0].astype(np.float64, copy=True)
    out = np.empty_like(iq_data)
    for t in range(iq_data.shape[0]):
        current = iq_data[t].astype(np.float64)
        out[t] = (current - bg).astype(iq_data.dtype)
        bg = alpha * current + (1.0 - alpha) * bg
    return out


def preprocess_window(window: np.ndarray, stats: dict) -> np.ndarray:
    complex_iq = window[..., 0] + 1j * window[..., 1]
    fft_vals = np.fft.fft(complex_iq, axis=0)

    fft_real = np.real(fft_vals).astype(np.float64).transpose(1, 2, 3, 0)
    fft_imag = np.imag(fft_vals).astype(np.float64).transpose(1, 2, 3, 0)

    fft_real = fft_real[
        :,
        :,
        stats["fft_bin_lo"] : stats["fft_bin_hi"] + 1,
        stats["fft_freq_lo"] : stats["fft_freq_hi"] + 1,
    ]
    fft_imag = fft_imag[
        :,
        :,
        stats["fft_bin_lo"] : stats["fft_bin_hi"] + 1,
        stats["fft_freq_lo"] : stats["fft_freq_hi"] + 1,
    ]

    fft_amp = np.sqrt(fft_real**2 + fft_imag**2)
    features = np.stack([fft_real, fft_imag, fft_amp], axis=-1).astype(np.float32)
    return standardize_features(features, stats)


def standardize_features(features: np.ndarray, stats: dict) -> np.ndarray:
    mean = stats["mean"].astype(np.float32)
    denominator = stats["denominator"].astype(np.float32)
    return ((features - mean) / denominator).astype(np.float32)


def _np_scalar_to_str(value):
    arr = np.asarray(value)
    item = arr.item() if arr.shape == () else arr
    if isinstance(item, bytes):
        return item.decode("utf-8")
    return str(item)


def _load_tflite_schema():
    schema_path = Path(tf.__file__).parent / "lite" / "python" / "schema_py_generated.py"
    if not schema_path.exists():
        raise RuntimeError(f"TFLite schema not found at {schema_path}")
    spec = importlib.util.spec_from_file_location("tflite_schema", schema_path)
    schema = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(schema)
    return schema


def _read_tflite_metadata(model_path: Path, metadata_name: bytes) -> bytes:
    schema = _load_tflite_schema()
    model_buf = model_path.read_bytes()
    model_obj = schema.Model.GetRootAs(model_buf, 0)
    target_name = metadata_name.decode("utf-8")

    for i in range(model_obj.MetadataLength()):
        metadata = model_obj.Metadata(i)
        name = metadata.Name()
        name_text = name.decode("utf-8") if isinstance(name, bytes) else str(name)
        if name_text != target_name:
            continue

        buffer_obj = model_obj.Buffers(metadata.Buffer())
        data = buffer_obj.DataAsNumpy()
        if isinstance(data, int):
            raise RuntimeError(f"TFLite metadata {target_name} has empty data")
        return bytes(np.asarray(data, dtype=np.uint8))

    raise RuntimeError(f"TFLite metadata {target_name} not found in {model_path}")


def load_stats_from_tflite(model_path: Path) -> dict:
    payload = _read_tflite_metadata(model_path, TFLITE_MODEL_METADATA_NAME)
    raw = np.load(io.BytesIO(payload), allow_pickle=False)
    mode = _np_scalar_to_str(raw["normalization_stats_mode"])
    if mode != "share_radars":
        raise RuntimeError(f"Unsupported normalization_stats_mode in TFLite: {mode}")

    return {
        "fft_bin_lo": FFT_BIN_LO,
        "fft_bin_hi": FFT_BIN_HI,
        "fft_freq_lo": FFT_FREQ_LO,
        "fft_freq_hi": FFT_FREQ_HI,
        "sub_window_size": SUB_WINDOW_SIZE,
        "thresholds": raw["thresholds"].astype(np.float32),
        "background_mode": BACKGROUND_MODE,
        "ema_alpha": EMA_ALPHA,
        "mean": raw["mean"],
        "denominator": raw["denominator"],
    }


def _dequantized_tensor(interpreter, output_detail):
    raw = interpreter.get_tensor(output_detail["index"]).flatten()
    scale, zero_point = output_detail["quantization"]
    if scale == 0:
        return raw.astype(np.float32)
    return (raw.astype(np.float32) - zero_point) * scale


def _quantize_input(features, input_detail):
    dtype = input_detail["dtype"]
    if np.issubdtype(dtype, np.floating):
        return features.astype(dtype)

    scale, zero_point = input_detail["quantization"]
    if scale == 0:
        raise ValueError("Quantized model input has zero scale")

    quantized = np.round(features / scale + zero_point)
    info = np.iinfo(dtype)
    return np.clip(quantized, info.min, info.max).astype(dtype)


def _sigmoid(x):
    x = np.clip(x, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-x))


def _slot_sort_key(output_detail):
    name = output_detail["name"].lower()
    matches = re.findall(r"(\d+)", name)
    if matches:
        return int(matches[-1])
    return name


def _all_have_numeric_suffix(output_details):
    return all(re.findall(r"(\d+)", o["name"].lower()) for o in output_details)


def _output_groups(output_details):
    named_xy = [o for o in output_details if "xy" in o["name"].lower()]
    named_pr = [o for o in output_details if "presence" in o["name"].lower()]

    if len(named_xy) == N_SLOTS and len(named_pr) == N_SLOTS:
        return sorted(named_xy, key=_slot_sort_key), sorted(named_pr, key=_slot_sort_key)

    sortable = _all_have_numeric_suffix(output_details)
    ordered_outputs = (
        sorted(output_details, key=_slot_sort_key) if sortable else output_details
    )
    xy_out = [o for o in ordered_outputs if int(np.prod(o["shape"])) == 2]
    pr_out = [o for o in ordered_outputs if int(np.prod(o["shape"])) == 1]

    if len(xy_out) != N_SLOTS or len(pr_out) != N_SLOTS:
        details = ", ".join(
            f"{o['name']} shape={tuple(int(x) for x in o['shape'])}"
            for o in output_details
        )
        raise RuntimeError(
            f"Could not identify {N_SLOTS} xy and {N_SLOTS} presence outputs. "
            f"Model outputs: {details}"
        )

    return xy_out, pr_out


def _check_input_shape(features, input_detail):
    expected = tuple(int(x) for x in input_detail["shape"][1:])
    if expected and all(dim > 0 for dim in expected) and features.shape != expected:
        raise RuntimeError(
            f"Preprocessed feature shape {features.shape} does not match "
            f"model input shape {expected}."
        )


def run_inference(interpreter, input_details, output_details, raw_data, stats):
    sub_window_size = stats["sub_window_size"]
    thresholds = np.asarray(stats["thresholds"], dtype=np.float32)

    # Match training preprocessing: keep one EMA state for the full acquisition,
    # then extract the already-filtered sliding windows used by the model.
    filtered_data = raw_data
    if stats["background_mode"] == "ema":
        filtered_data = apply_ema_iq(raw_data, stats["ema_alpha"])

    input_detail = input_details[0]
    xy_out, pr_out = _output_groups(output_details)

    predictions = []
    for t in range(raw_data.shape[0]):
        start = max(0, t - sub_window_size + 1)
        window = filtered_data[start : t + 1]

        if window.shape[0] < sub_window_size:
            pad_shape = (sub_window_size - window.shape[0],) + window.shape[1:]
            pad = np.zeros(pad_shape, dtype=raw_data.dtype)
            window = np.concatenate([pad, window], axis=0)

        features = preprocess_window(window, stats)
        _check_input_shape(features, input_detail)
        model_input = _quantize_input(features[None, ...], input_detail)

        interpreter.set_tensor(input_detail["index"], model_input)
        interpreter.invoke()

        locs = []
        for slot, (xy_det, pr_det) in enumerate(zip(xy_out, pr_out)):
            xy = _dequantized_tensor(interpreter, xy_det)
            pr = float(_dequantized_tensor(interpreter, pr_det)[0])
            if _sigmoid(pr) > thresholds[slot]:
                x = max(0.0, min(float(xy[0]), ROOM_X_MAX))
                y = max(0.0, min(float(xy[1]), ROOM_Y_MAX))
                locs.append([x, y])

        predictions.append({"frame": t, "localizations": locs})

    if raw_data.shape[0] > sub_window_size - 1:
        first_full_locs = predictions[sub_window_size - 1]["localizations"].copy()
        for t in range(min(sub_window_size - 1, raw_data.shape[0])):
            predictions[t]["localizations"] = first_full_locs.copy()

    return predictions


def main():
    args = parse_args()
    input_path = Path(args.input_path)
    output_path = Path(args.output_path)
    model_dir = Path(args.model_dir) if args.model_dir else Path(__file__).parent

    if not input_path.exists():
        print(f"[ERROR] Input not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    model_path = Path(args.model_path) if args.model_path else model_dir / "model.tflite"
    if not model_path.exists():
        print(f"[ERROR] model not found: {model_path}", file=sys.stderr)
        sys.exit(1)

    interpreter = tf.lite.Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()

    raw_data = np.load(input_path)
    stats = load_stats_from_tflite(model_path)
    predictions = run_inference(
        interpreter,
        interpreter.get_input_details(),
        interpreter.get_output_details(),
        raw_data,
        stats,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for pred in predictions:
            f.write(json.dumps(pred) + "\n")

    print(f"Saved {len(predictions)} predictions to {output_path}")


if __name__ == "__main__":
    main()
