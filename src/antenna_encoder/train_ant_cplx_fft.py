import os
import glob
import csv
import itertools
import json
import numpy as np
import tensorflow as tf
from scipy.optimize import linear_sum_assignment
from scipy.special import expit

# ==========================================================================
# Hyperparameters (from environment variables)
# ==========================================================================
SEED = int(os.environ.get("SEED", 42))
SUB_WINDOW_SIZE = int(os.environ.get("SUB_WINDOW_SIZE", 50))
MACRO_WINDOW_SPLIT = int(os.environ.get("MACRO_WINDOW_SPLIT", 1))
TRAIN_RATIO = float(os.environ.get("TRAIN_RATIO", 0.7))
VAL_RATIO = float(os.environ.get("VAL_RATIO", 0.15))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 32))
EPOCHS = int(os.environ.get("EPOCHS", 10))
LEARNING_RATE = float(os.environ.get("LEARNING_RATE", 1e-3))
WEIGHT_DECAY = float(os.environ.get("WEIGHT_DECAY", 0.0))
DROPOUT = float(os.environ.get("DROPOUT", 0.0))
DATA_DIR = os.environ.get("DATA_DIR", "dataset/data")
STD_EPSILON = float(os.environ.get("STD_EPSILON", 0.0))
BASELINE_ZERO_PEOPLE = bool(int(os.environ.get("BASELINE_ZERO_PEOPLE", 0)))
BACKGROUND_MODE = os.environ.get("BACKGROUND_MODE", None)
EMA_ALPHA = float(os.environ.get("EMA_ALPHA", "0.1"))
if BACKGROUND_MODE is None:
    BACKGROUND_MODE = "baseline" if BASELINE_ZERO_PEOPLE else "none"
HUNGARIAN_LOSS = bool(int(os.environ.get("HUNGARIAN_LOSS", 0)))
EXPERIMENT_NAME = os.environ.get("EXPERIMENT_NAME", "default")
CONV_FILTERS = os.environ.get("CONV_FILTERS", "32,48")
CONV_KERNELS = os.environ.get("CONV_KERNELS", "5,3")
D_ATTN = int(os.environ.get("D_ATTN", 16))
N_HEADS = int(os.environ.get("N_HEADS", 1))
DENSE_DIM = int(os.environ.get("DENSE_DIM", 64))
HEAD_DIM = int(os.environ.get("HEAD_DIM", 16))
POS_EMBED = bool(int(os.environ.get("POS_EMBED", 0)))
ANTENNA_D_PROJ = int(os.environ.get("ANTENNA_D_PROJ", 32))
FFT_BIN_RANGE = os.environ.get("FFT_BIN_RANGE", "0,119")
FFT_FREQ_RANGE = os.environ.get("FFT_FREQ_RANGE", "0,25")
LR_SCHEDULE = os.environ.get("LR_SCHEDULE", "none")
WARMUP_EPOCHS = int(os.environ.get("WARMUP_EPOCHS", 5))
TRANS_BLOCKS = int(os.environ.get("TRANS_BLOCKS", 1))
FFN_MULT = float(os.environ.get("FFN_MULT", 4))
NORM_TYPE = str(os.environ.get("NORM_TYPE", "batch"))
ENCODER_POOLING = os.environ.get("ENCODER_POOLING", "gap")
ENCODER_TOPK = int(os.environ.get("ENCODER_TOPK", 8))
ENCODER_ATTN_TEMPERATURE = float(os.environ.get("ENCODER_ATTN_TEMPERATURE", 4.0))
USE_FOCAL_LOSS = bool(int(os.environ.get("USE_FOCAL_LOSS", 0)))
AMPLITUDE_CHANNEL = bool(int(os.environ.get("AMPLITUDE_CHANNEL", 0)))
ANTENNA_DROPOUT = float(os.environ.get("ANTENNA_DROPOUT", 0.0))
PHASE_SHIFT_MAX = float(os.environ.get("PHASE_SHIFT_MAX", 0.0))
MASK_LOSS_WEIGHT = float(os.environ.get("MASK_LOSS_WEIGHT", 1.0))
CLASS_BALANCE = bool(int(os.environ.get("CLASS_BALANCE", 0)))
CLASS_LOSS_SCALE = bool(int(os.environ.get("CLASS_LOSS_SCALE", 0)))
APPLY_NMS = bool(int(os.environ.get("APPLY_NMS", 0)))
COUNT_HEAD_WEIGHT = float(os.environ.get("COUNT_HEAD_WEIGHT", "0.0"))

fft_bin_lo, fft_bin_hi = [int(x) for x in FFT_BIN_RANGE.split(",")]
fft_freq_lo, fft_freq_hi = [int(x) for x in FFT_FREQ_RANGE.split(",")]

# ==========================================================================
# Reproducibility
# ==========================================================================
np.random.seed(SEED)
tf.random.set_seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)

# ==========================================================================
# Model definition: per-radar-antenna Conv2D + antenna attention + radar attention + transformer blocks + 4 heads
# ==========================================================================
N_RADARS = 6
N_ANTENNAS = 3
N_BINS = 120
N_FREQ = SUB_WINDOW_SIZE

n_bins_kept = fft_bin_hi - fft_bin_lo + 1
n_freq_kept = fft_freq_hi - fft_freq_lo + 1
n_channels = 3 if AMPLITUDE_CHANNEL else 2

conv_filters = [int(x) for x in CONV_FILTERS.split(",")]
conv_kernels = [int(x) for x in CONV_KERNELS.split(",")]
assert len(conv_filters) == len(conv_kernels), (
    "CONV_FILTERS and CONV_KERNELS must have same length"
)
assert fft_bin_lo >= 0 and fft_bin_hi < N_BINS, (
    f"FFT_BIN_RANGE must be within [0, {N_BINS - 1}]"
)
assert fft_freq_lo >= 0 and fft_freq_hi < N_FREQ, (
    f"FFT_FREQ_RANGE must be within [0, {N_FREQ - 1}]"
)
assert fft_bin_lo <= fft_bin_hi, "FFT_BIN_RANGE lo must be <= hi"
assert fft_freq_lo <= fft_freq_hi, "FFT_FREQ_RANGE lo must be <= hi"

if BACKGROUND_MODE not in ("none", "baseline", "ema"):
    raise ValueError(
        f"BACKGROUND_MODE must be none, baseline, or ema, got: {BACKGROUND_MODE}"
    )
if BACKGROUND_MODE == "ema" and not (0.0 < EMA_ALPHA <= 1.0):
    raise ValueError(f"EMA_ALPHA must be in (0, 1], got: {EMA_ALPHA}")
if ENCODER_POOLING not in (
    "gap",
    "topk",
    "topk8",
    "topk16",
    "gap_topk",
    "gap_topk8",
    "soft_attn",
    "soft_attn_temp4",
):
    raise ValueError(f"Unsupported ENCODER_POOLING: {ENCODER_POOLING}")
if ENCODER_TOPK <= 0:
    raise ValueError(f"ENCODER_TOPK must be > 0, got: {ENCODER_TOPK}")
if ENCODER_ATTN_TEMPERATURE <= 0.0:
    raise ValueError(
        f"ENCODER_ATTN_TEMPERATURE must be > 0, got: {ENCODER_ATTN_TEMPERATURE}"
    )


def apply_ema_iq(iq_data, alpha):
    iq_f = iq_data.astype(np.float64)
    bg = iq_f[0].copy()
    out = np.empty_like(iq_f)
    for t in range(iq_f.shape[0]):
        out[t] = iq_f[t] - bg
        bg = alpha * iq_f[t] + (1.0 - alpha) * bg
    return out.astype(iq_data.dtype)


def _focal_bce(labels, logits, gamma=2.0, alpha=0.25):
    probs = tf.sigmoid(logits)
    p_t = labels * probs + (1 - labels) * (1 - probs)
    alpha_t = labels * alpha + (1 - labels) * (1 - alpha)
    ce = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits)
    return alpha_t * tf.pow(1.0 - p_t, gamma) * ce


def _mask_bce_per_sample(labels, logits):
    if USE_FOCAL_LOSS:
        return _focal_bce(labels, logits)
    return tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits)


def _mask_bce(labels, logits, weights=None):
    per_sample = _mask_bce_per_sample(labels, logits)
    if weights is None:
        return tf.reduce_mean(per_sample)
    w = tf.cast(weights, per_sample.dtype)
    return tf.reduce_sum(per_sample * w) / (tf.reduce_sum(w) + 1e-8)


@tf.keras.utils.register_keras_serializable()
class AntennaDropout(tf.keras.layers.Layer):
    def __init__(self, rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.rate = rate

    def get_config(self):
        config = super().get_config()
        config.update({"rate": self.rate})
        return config

    def call(self, inputs, training=None):
        if not training or self.rate == 0.0:
            return inputs
        B = tf.shape(inputs)[0]
        keep_prob = 1.0 - self.rate
        mask = tf.random.uniform([B, N_RADARS, N_ANTENNAS, 1, 1, 1]) < keep_prob
        mask = tf.cast(mask, inputs.dtype)
        return inputs * mask * (1.0 / keep_prob)


@tf.keras.utils.register_keras_serializable()
class RandomPhaseShift(tf.keras.layers.Layer):
    def __init__(self, max_shift=0.5, **kwargs):
        super().__init__(**kwargs)
        self.max_shift = max_shift

    def get_config(self):
        config = super().get_config()
        config.update({"max_shift": self.max_shift})
        return config

    def call(self, inputs, training=None):
        if not training or self.max_shift == 0.0:
            return inputs
        B = tf.shape(inputs)[0]
        theta = tf.random.uniform([B, 1, 1, 1, 1, 1], -self.max_shift, self.max_shift)
        cos_t = tf.cos(theta)
        sin_t = tf.sin(theta)
        real = inputs[..., 0:1]
        imag = inputs[..., 1:2]
        new_real = real * cos_t - imag * sin_t
        new_imag = real * sin_t + imag * cos_t
        if n_channels == 3:
            amp = inputs[..., 2:3]
            return tf.concat([new_real, new_imag, amp], axis=-1)
        return tf.concat([new_real, new_imag], axis=-1)


@tf.keras.utils.register_keras_serializable()
class AntennaAttention(tf.keras.layers.Layer):
    def __init__(self, d_k, n_heads, **kwargs):
        super().__init__(**kwargs)
        self.d_k = d_k
        self.n_heads = n_heads
        self.query_proj = tf.keras.layers.Dense(d_k * n_heads)
        self.key_proj = tf.keras.layers.Dense(d_k * n_heads)
        self.value_proj = tf.keras.layers.Dense(d_k * n_heads)
        self.out_proj = tf.keras.layers.Dense(d_k * n_heads)

    def get_config(self):
        config = super().get_config()
        config.update({"d_k": self.d_k, "n_heads": self.n_heads})
        return config

    def call(self, x):
        B = tf.shape(x)[0]
        S = tf.shape(x)[1]

        Q = self.query_proj(x)
        K = self.key_proj(x)
        V = self.value_proj(x)

        Q = tf.reshape(Q, [B, S, self.n_heads, self.d_k])
        K = tf.reshape(K, [B, S, self.n_heads, self.d_k])
        V = tf.reshape(V, [B, S, self.n_heads, self.d_k])

        Q = tf.transpose(Q, [0, 2, 1, 3])
        K = tf.transpose(K, [0, 2, 1, 3])
        V = tf.transpose(V, [0, 2, 1, 3])

        scores = tf.matmul(Q, K, transpose_b=True) / tf.sqrt(tf.cast(self.d_k, x.dtype))
        weights = tf.nn.softmax(scores, axis=-1)
        attended = tf.matmul(weights, V)

        attended = tf.transpose(attended, [0, 2, 1, 3])
        attended = tf.reshape(attended, [B, S, self.n_heads * self.d_k])
        return self.out_proj(attended)


@tf.keras.utils.register_keras_serializable()
class MultiHeadRadarAttention(tf.keras.layers.Layer):
    def __init__(self, d_k, n_heads, **kwargs):
        super().__init__(**kwargs)
        self.d_k = d_k
        self.n_heads = n_heads
        self.query_proj = tf.keras.layers.Dense(d_k * n_heads)
        self.key_proj = tf.keras.layers.Dense(d_k * n_heads)
        self.value_proj = tf.keras.layers.Dense(d_k * n_heads)
        self.out_proj = tf.keras.layers.Dense(d_k * n_heads)

    def get_config(self):
        config = super().get_config()
        config.update({"d_k": self.d_k, "n_heads": self.n_heads})
        return config

    def call(self, x):
        B = tf.shape(x)[0]
        S = tf.shape(x)[1]

        Q = self.query_proj(x)
        K = self.key_proj(x)
        V = self.value_proj(x)

        Q = tf.reshape(Q, [B, S, self.n_heads, self.d_k])
        K = tf.reshape(K, [B, S, self.n_heads, self.d_k])
        V = tf.reshape(V, [B, S, self.n_heads, self.d_k])

        Q = tf.transpose(Q, [0, 2, 1, 3])
        K = tf.transpose(K, [0, 2, 1, 3])
        V = tf.transpose(V, [0, 2, 1, 3])

        scores = tf.matmul(Q, K, transpose_b=True) / tf.sqrt(tf.cast(self.d_k, x.dtype))
        weights = tf.nn.softmax(scores, axis=-1)
        attended = tf.matmul(weights, V)

        attended = tf.transpose(attended, [0, 2, 1, 3])
        attended = tf.reshape(attended, [B, S, self.n_heads * self.d_k])
        return self.out_proj(attended)


def norm():
    return (
        tf.keras.layers.BatchNormalization()
        if NORM_TYPE == "batch"
        else tf.keras.layers.LayerNormalization()
    )


@tf.keras.utils.register_keras_serializable()
class TopKAveragePooling2D(tf.keras.layers.Layer):
    def __init__(self, k=8, **kwargs):
        super().__init__(**kwargs)
        self.k = k

    def get_config(self):
        config = super().get_config()
        config.update({"k": self.k})
        return config

    def call(self, inputs):
        shape = tf.shape(inputs)
        flat = tf.reshape(inputs, [shape[0], -1, shape[-1]])
        flat = tf.transpose(flat, [0, 2, 1])
        k = tf.minimum(tf.cast(self.k, tf.int32), tf.shape(flat)[-1])
        values = tf.nn.top_k(flat, k=k, sorted=False).values
        return tf.reduce_mean(values, axis=-1)


@tf.keras.utils.register_keras_serializable()
class SoftAttentionPooling2D(tf.keras.layers.Layer):
    def __init__(self, temperature=4.0, **kwargs):
        super().__init__(**kwargs)
        self.temperature = temperature
        self.score_proj = tf.keras.layers.Conv2D(1, kernel_size=1, padding="same")

    def get_config(self):
        config = super().get_config()
        config.update({"temperature": self.temperature})
        return config

    def call(self, inputs):
        shape = tf.shape(inputs)
        scores = self.score_proj(inputs) / self.temperature
        scores = tf.reshape(scores, [shape[0], -1, 1])
        weights = tf.nn.softmax(scores, axis=1)
        flat = tf.reshape(inputs, [shape[0], -1, shape[-1]])
        return tf.reduce_sum(flat * weights, axis=1)


def _topk_for_pooling():
    if ENCODER_POOLING in ("topk8", "gap_topk8"):
        return 8
    if ENCODER_POOLING == "topk16":
        return 16
    return ENCODER_TOPK


def encoder_pool(x, name):
    if ENCODER_POOLING == "gap":
        return tf.keras.layers.GlobalAveragePooling2D(name=f"{name}_gap")(x)
    if ENCODER_POOLING in ("topk", "topk8", "topk16"):
        return TopKAveragePooling2D(_topk_for_pooling(), name=f"{name}_topk")(x)
    if ENCODER_POOLING in ("gap_topk", "gap_topk8"):
        gap = tf.keras.layers.GlobalAveragePooling2D(name=f"{name}_gap")(x)
        topk = TopKAveragePooling2D(_topk_for_pooling(), name=f"{name}_topk")(x)
        return tf.keras.layers.Concatenate(name=f"{name}_gap_topk")([gap, topk])
    if ENCODER_POOLING in ("soft_attn", "soft_attn_temp4"):
        temperature = (
            4.0 if ENCODER_POOLING == "soft_attn_temp4" else ENCODER_ATTN_TEMPERATURE
        )
        return SoftAttentionPooling2D(temperature, name=f"{name}_soft_attn")(x)
    raise ValueError(f"Unsupported ENCODER_POOLING: {ENCODER_POOLING}")


inp = tf.keras.Input(
    shape=(N_RADARS, N_ANTENNAS, n_bins_kept, n_freq_kept, n_channels), name="input"
)

x = inp
if PHASE_SHIFT_MAX > 0.0:
    x = RandomPhaseShift(PHASE_SHIFT_MAX)(x)
if ANTENNA_DROPOUT > 0.0:
    x = AntennaDropout(ANTENNA_DROPOUT)(x)

# Process each radar-antenna combination independently
x = tf.keras.layers.Lambda(
    lambda t: tf.reshape(t, [-1, n_bins_kept, n_freq_kept, n_channels])
)(x)

for i, (f, k) in enumerate(zip(conv_filters, conv_kernels)):
    x = tf.keras.layers.Conv2D(f, kernel_size=(k, k), padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(x)

last_conv_dim = conv_filters[-1]
pooled_dim = last_conv_dim * 2 if ENCODER_POOLING in ("gap_topk", "gap_topk8") else last_conv_dim

x = encoder_pool(x, "encoder")
x = tf.keras.layers.Lambda(
    lambda t: tf.reshape(t, [-1, N_RADARS, N_ANTENNAS, pooled_dim])
)(x)

# Antenna-level attention
x = tf.keras.layers.Lambda(
    lambda t: tf.reshape(t, [-1, N_ANTENNAS, pooled_dim])
)(x)
x = AntennaAttention(D_ATTN, N_HEADS)(x)
x = tf.keras.layers.Reshape((N_ANTENNAS * D_ATTN * N_HEADS,))(x)
x = tf.keras.layers.Dense(ANTENNA_D_PROJ)(x)
x = tf.keras.layers.Lambda(
    lambda t: tf.reshape(t, [-1, N_RADARS, ANTENNA_D_PROJ])
)(x)

# Radar-level attention
if POS_EMBED:
    radar_pe = tf.Variable(
        tf.random.normal([1, N_RADARS, ANTENNA_D_PROJ], stddev=0.02),
        trainable=True,
        name="radar_pos_embed",
    )
    x = tf.keras.layers.Lambda(lambda t: t + radar_pe)(x)

x = MultiHeadRadarAttention(D_ATTN, N_HEADS)(x)

# Transformer blocks after radar attention
d_model = D_ATTN * N_HEADS
for i in range(TRANS_BLOCKS):
    residual = x
    x = norm()(x)
    x = MultiHeadRadarAttention(D_ATTN, N_HEADS)(x)
    x = tf.keras.layers.Add()([x, residual])

    if i < TRANS_BLOCKS - 1:
        residual = x
        x = norm()(x)
        x = tf.keras.layers.Dense(int(d_model * FFN_MULT), activation="relu")(x)
        x = tf.keras.layers.Dense(d_model)(x)
        x = tf.keras.layers.Add()([x, residual])

x = tf.keras.layers.Reshape((N_RADARS * d_model,))(x)

x = tf.keras.layers.Dense(DENSE_DIM)(x)
x = norm()(x)
x = tf.keras.layers.ReLU()(x)
x = tf.keras.layers.Dropout(DROPOUT)(x)

outputs = []
for i in range(4):
    hi = tf.keras.layers.Dense(HEAD_DIM, activation="relu")(x)
    xy_i = tf.keras.layers.Dense(2, name=f"p{i}_xy")(hi)
    presence_i = tf.keras.layers.Dense(1, name=f"p{i}_presence")(hi)
    outputs.extend([xy_i, presence_i])

if COUNT_HEAD_WEIGHT > 0.0:
    count_logits = tf.keras.layers.Dense(5, name="count_logits")(x)
    outputs.append(count_logits)

model = tf.keras.Model(inputs=inp, outputs=outputs)
model.summary()

print(f"\n=== Layer breakdown ===")
print(f"{'Layer':<40s} {'Output Shape':<25s} {'Params':>10s}")
print("-" * 78)
total = 0
trainable = 0
for layer in model.layers:
    params = layer.count_params()
    tr_p = sum(np.prod(v.shape) for v in layer.trainable_variables)
    out_shape = str(layer.output_shape) if hasattr(layer, "output_shape") else "?"
    if params > 0:
        print(f"{layer.name:<40s} {out_shape:<25s} {params:>10d}")
    total += params
    trainable += tr_p
print("-" * 78)
print(f"{'Total':<40s} {'':<25s} {total:>10d}")
print(f"{'Trainable':<40s} {'':<25s} {trainable:>10d}")
print(f"{'Model size (float32)':<40s} {'':<25s} {total * 4 / 1024:>8.1f} KB")
print(f"{'Model size (int8 est.)':<40s} {'':<25s} {total / 1024:>8.1f} KB")
print()

print(f"experiment: {EXPERIMENT_NAME}")

# ==========================================================================
# Data loading and preprocessing
# ==========================================================================
files = sorted(glob.glob(os.path.join(DATA_DIR, "window_*.npz")))
print(f"Found {len(files)} windows")

macro_features = []
macro_target_xy = []
macro_target_mask = []
macro_sources = []

for fpath in files:
    d = np.load(fpath)
    iq = d["radar_cir_iq"]
    if BACKGROUND_MODE == "ema":
        iq = apply_ema_iq(iq, EMA_ALPHA)
    people_xy = d["people_xy"]
    people_mask = d["people_mask"]

    T = iq.shape[0]
    chunk_size = SUB_WINDOW_SIZE
    n_sub = T // chunk_size
    assert n_sub % MACRO_WINDOW_SPLIT == 0, (
        f"n_sub={n_sub} not divisible by MACRO_WINDOW_SPLIT={MACRO_WINDOW_SPLIT}"
    )
    macro_size = n_sub // MACRO_WINDOW_SPLIT
    usable = n_sub * chunk_size

    iq = iq[:usable]
    people_xy = people_xy[:usable]
    people_mask = people_mask[:usable]

    iq_sub = iq.reshape(n_sub, chunk_size, 6, 3, 120, 2)
    xy_sub = people_xy.reshape(n_sub, chunk_size, 4, 2)
    mask_sub = people_mask.reshape(n_sub, chunk_size, 4)

    for m_idx in range(MACRO_WINDOW_SPLIT):
        sl = slice(m_idx * macro_size, (m_idx + 1) * macro_size)
        iq_m = iq_sub[sl]
        xy_m = xy_sub[sl]
        mask_m = mask_sub[sl]

        complex_iq = iq_m[..., 0].astype(np.float64) + 1j * iq_m[..., 1].astype(
            np.float64
        )
        fft_vals = np.fft.fft(complex_iq, axis=1)
        fft_real = np.real(fft_vals).astype(np.float64)
        fft_imag = np.imag(fft_vals).astype(np.float64)
        fft_real = fft_real.transpose(0, 2, 3, 4, 1)
        fft_imag = fft_imag.transpose(0, 2, 3, 4, 1)
        fft_real = fft_real[
            :, :, :, fft_bin_lo : fft_bin_hi + 1, fft_freq_lo : fft_freq_hi + 1
        ]
        fft_imag = fft_imag[
            :, :, :, fft_bin_lo : fft_bin_hi + 1, fft_freq_lo : fft_freq_hi + 1
        ]
        if AMPLITUDE_CHANNEL:
            fft_amp = np.sqrt(fft_real**2 + fft_imag**2)
            features = np.stack([fft_real, fft_imag, fft_amp], axis=-1).astype(
                np.float32
            )
        else:
            features = np.stack([fft_real, fft_imag], axis=-1).astype(np.float32)
        target_xy = xy_m[:, -1, :, :].astype(np.float32)
        target_mask = mask_m[:, -1, :].astype(np.float32)

        macro_features.append(features)
        macro_target_xy.append(target_xy)
        macro_target_mask.append(target_mask)

        macro_sources.append({
            "file_path": fpath,
            "file_idx": int(os.path.basename(fpath).split("_")[-1].split(".")[0]),
            "macro_idx": m_idx,
            "start_frame": int(m_idx * macro_size * chunk_size),
            "end_frame": int((m_idx + 1) * macro_size * chunk_size - 1),
            "n_frames": int(macro_size * chunk_size),
        })

n_macros = len(macro_features)
print(f"Total macro-windows: {n_macros}")

# ==========================================================================
# Stratified shuffle and train/val/test split (at macro-window level)
# ==========================================================================
macro_people_count = [int(macro_target_mask[i][0].sum()) for i in range(n_macros)]

by_count = {k: [] for k in range(5)}
for i in range(n_macros):
    by_count[macro_people_count[i]].append(i)

train_macro, val_macro, test_macro = [], [], []
for k in range(5):
    indices_k = np.array(by_count[k])
    np.random.shuffle(indices_k)
    n_k = len(indices_k)
    t_end = int(n_k * TRAIN_RATIO)
    v_end = int(n_k * (TRAIN_RATIO + VAL_RATIO))
    train_macro.extend(indices_k[:t_end].tolist())
    val_macro.extend(indices_k[t_end:v_end].tolist())
    test_macro.extend(indices_k[v_end:].tolist())

train_features = np.concatenate([macro_features[i] for i in train_macro], axis=0)
train_xy = np.concatenate([macro_target_xy[i] for i in train_macro], axis=0)
train_mask = np.concatenate([macro_target_mask[i] for i in train_macro], axis=0)
val_features = np.concatenate([macro_features[i] for i in val_macro], axis=0)
val_xy = np.concatenate([macro_target_xy[i] for i in val_macro], axis=0)
val_mask = np.concatenate([macro_target_mask[i] for i in val_macro], axis=0)
test_features = np.concatenate([macro_features[i] for i in test_macro], axis=0)
test_xy = np.concatenate([macro_target_xy[i] for i in test_macro], axis=0)
test_mask = np.concatenate([macro_target_mask[i] for i in test_macro], axis=0)

N = train_features.shape[0] + val_features.shape[0] + test_features.shape[0]
print(
    f"Split — train: {train_features.shape[0]}, val: {val_features.shape[0]}, "
    f"test: {test_features.shape[0]}  (total sub-windows: {N})"
)

# ==========================================================================
# Class balancing (oversample under-represented count classes in train set)
# ==========================================================================
if CLASS_BALANCE:
    train_counts = (train_mask > 0.5).sum(axis=1).astype(int)
    by_count_idx = {k: np.where(train_counts == k)[0] for k in range(5)}
    nonempty = {k: idx for k, idx in by_count_idx.items() if len(idx) > 0}
    max_n = max(len(idx) for idx in nonempty.values())
    rng = np.random.RandomState(SEED)
    rebalanced_idx = []
    for k, idx in nonempty.items():
        repeats = int(np.ceil(max_n / len(idx)))
        rep = np.tile(idx, repeats)
        rng.shuffle(rep)
        rebalanced_idx.append(rep[:max_n])
    rebalanced_idx = np.concatenate(rebalanced_idx)
    rng.shuffle(rebalanced_idx)
    print(
        f"Class balancing: oversampled to "
        f"{len(rebalanced_idx)} samples ({max_n} per non-empty class)"
    )
    train_features = train_features[rebalanced_idx]
    train_xy = train_xy[rebalanced_idx]
    train_mask = train_mask[rebalanced_idx]

# ==========================================================================
# Per-sample loss weights (inverse class frequency)
# ==========================================================================
train_counts = (train_mask > 0.5).sum(axis=1).astype(int)
if CLASS_LOSS_SCALE:
    class_counts = np.bincount(train_counts, minlength=5).astype(np.float64)
    nonzero = class_counts > 0
    n_classes = int(nonzero.sum())
    class_weights_lookup = np.zeros(5, dtype=np.float64)
    class_weights_lookup[nonzero] = (
        len(train_counts) / (n_classes * class_counts[nonzero])
    )
    train_weights = class_weights_lookup[train_counts].astype(np.float32)
    print(
        f"Class loss scaling weights (per true count): "
        f"{ {k: float(class_weights_lookup[k]) for k in range(5)} }"
    )
else:
    train_weights = np.ones(len(train_features), dtype=np.float32)
val_weights = np.ones(len(val_features), dtype=np.float32)
test_weights = np.ones(len(test_features), dtype=np.float32)

# ==========================================================================
# Background subtraction
# ==========================================================================
if BACKGROUND_MODE == "baseline":
    zero_mask = train_mask.sum(axis=1) == 0
    zero_train_idx = np.where(zero_mask)[0]
    if len(zero_train_idx) == 0:
        print("WARNING: no 0-people sub-windows in train set, skipping baseline")
    else:
        zero_feats = train_features[zero_train_idx]
        bl_med = np.median(zero_feats, axis=0)
        train_features -= bl_med
        val_features -= bl_med
        test_features -= bl_med
        print(
            f"Background subtracted (baseline) from {len(zero_train_idx)} 0-people train sub-windows"
        )

# ==========================================================================
# Standardization (mean/std per radar, antenna, bin, freq, channel — computed on train set)
# ==========================================================================
feat_mean = train_features.mean(axis=0)
feat_std = train_features.std(axis=0)
feat_std[feat_std == 0] = 1.0

train_features = (train_features - feat_mean) / (feat_std + STD_EPSILON)
val_features = (val_features - feat_mean) / (feat_std + STD_EPSILON)
test_features = (test_features - feat_mean) / (feat_std + STD_EPSILON)

# Save normalization stats and test metadata
model_dir = os.path.join(DATA_DIR, "exported_models", EXPERIMENT_NAME)
os.makedirs(model_dir, exist_ok=True)

test_sources = [macro_sources[i] for i in test_macro]
metadata = {
    "seed": SEED,
    "sub_window_size": SUB_WINDOW_SIZE,
    "macro_window_split": MACRO_WINDOW_SPLIT,
    "test_sources": test_sources,
    "n_test_macros": len(test_sources),
    "experiment_name": EXPERIMENT_NAME,
    "background_mode": BACKGROUND_MODE,
    "ema_alpha": EMA_ALPHA if BACKGROUND_MODE == "ema" else None,
}
with open(os.path.join(model_dir, "test_metadata.json"), "w") as f:
    json.dump(metadata, f, indent=2)
print(f"Saved test_metadata.json to {model_dir}/test_metadata.json")

# ==========================================================================
# tf.data Datasets
# ==========================================================================
_sample_spec = (
    tf.TensorSpec(
        shape=(N_RADARS, N_ANTENNAS, n_bins_kept, n_freq_kept, n_channels),
        dtype=tf.float32,
    ),
    {
        "xy": tf.TensorSpec(shape=(4, 2), dtype=tf.float32),
        "mask": tf.TensorSpec(shape=(4,), dtype=tf.float32),
        "weight": tf.TensorSpec(shape=(), dtype=tf.float32),
    },
)


def _make_generator(features, xy, mask, weights):
    def gen():
        for i in range(features.shape[0]):
            yield (
                features[i],
                {"xy": xy[i], "mask": mask[i], "weight": weights[i]},
            )

    return gen


train_ds = (
    tf.data.Dataset.from_generator(
        _make_generator(train_features, train_xy, train_mask, train_weights),
        output_signature=_sample_spec,
    )
    .shuffle(1024, seed=SEED)
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)
val_ds = (
    tf.data.Dataset.from_generator(
        _make_generator(val_features, val_xy, val_mask, val_weights),
        output_signature=_sample_spec,
    )
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)
test_ds = (
    tf.data.Dataset.from_generator(
        _make_generator(test_features, test_xy, test_mask, test_weights),
        output_signature=_sample_spec,
    )
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)


# ==========================================================================
# Loss and training loop
# ==========================================================================
steps_per_epoch = max(1, len(train_features) // BATCH_SIZE)
total_steps = EPOCHS * steps_per_epoch
warmup_steps = WARMUP_EPOCHS * steps_per_epoch
cosine_steps = total_steps - warmup_steps

if LR_SCHEDULE == "cosine":
    lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
        LEARNING_RATE, decay_steps=cosine_steps
    )
    if WARMUP_EPOCHS > 0:
        warmup_lr = tf.keras.optimizers.schedules.PolynomialDecay(
            initial_learning_rate=0.0,
            decay_steps=warmup_steps,
            end_learning_rate=LEARNING_RATE,
        )

        @tf.keras.utils.register_keras_serializable()
        class WarmupCosineSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
            def __init__(self, warmup_sched, cosine_sched, warmup_steps):
                super().__init__()
                self.warmup_sched = warmup_sched
                self.cosine_sched = cosine_sched
                self.warmup_steps = warmup_steps

            def get_config(self):
                return {
                    "warmup_sched": tf.keras.optimizers.schedules.serialize(
                        self.warmup_sched
                    ),
                    "cosine_sched": tf.keras.optimizers.schedules.serialize(
                        self.cosine_sched
                    ),
                    "warmup_steps": self.warmup_steps,
                }

            @classmethod
            def from_config(cls, config):
                return cls(
                    warmup_sched=tf.keras.optimizers.schedules.deserialize(
                        config["warmup_sched"]
                    ),
                    cosine_sched=tf.keras.optimizers.schedules.deserialize(
                        config["cosine_sched"]
                    ),
                    warmup_steps=config["warmup_steps"],
                )

            def __call__(self, step):
                return tf.cond(
                    step < self.warmup_steps,
                    lambda: self.warmup_sched(step),
                    lambda: self.cosine_sched(step - self.warmup_steps),
                )

        lr_schedule = WarmupCosineSchedule(warmup_lr, lr_schedule, warmup_steps)
    optimizer = tf.keras.optimizers.Adam(lr_schedule, weight_decay=WEIGHT_DECAY)
    print(
        f"LR schedule: cosine  warmup={WARMUP_EPOCHS} epochs ({warmup_steps} steps)  "
        f"cosine_decay={cosine_steps} steps  total={total_steps} steps"
    )
else:
    optimizer = tf.keras.optimizers.Adam(LEARNING_RATE, weight_decay=WEIGHT_DECAY)

_ALL_PERMS_4 = list(itertools.permutations(range(4)))
_PERMS_TF = tf.constant(_ALL_PERMS_4, dtype=tf.int32)


def _hungarian_4x4(cost_matrix):
    perm_costs = []
    for perm in _ALL_PERMS_4:
        c = sum(cost_matrix[:, i, perm[i]] for i in range(4))
        perm_costs.append(c)
    perm_costs = tf.stack(perm_costs, axis=1)
    best = tf.argmin(perm_costs, axis=1)
    return tf.gather(_PERMS_TF, best)


def _compute_fixed_loss(xy_true, mask_true, predictions, weights):
    xy_loss = tf.constant(0.0)
    mask_loss = tf.constant(0.0)
    w = tf.cast(weights, tf.float32)
    w_exp = tf.expand_dims(w, -1)
    for i in range(4):
        xy_pred = predictions[2 * i]
        presence_logits = predictions[2 * i + 1]
        m = mask_true[:, i]
        m_exp = tf.expand_dims(m, -1)
        sq_err = tf.square(xy_pred - xy_true[:, i, :]) * m_exp * w_exp
        n_valid = tf.reduce_sum(m * w) * 2.0 + 1e-8
        xy_loss += tf.reduce_sum(sq_err) / n_valid
        mask_loss += _mask_bce(m, presence_logits[:, 0], weights=w)
    loss = xy_loss + MASK_LOSS_WEIGHT * mask_loss
    if COUNT_HEAD_WEIGHT > 0.0:
        count_logits = predictions[-1]
        true_count = tf.cast(tf.reduce_sum(mask_true, axis=1), tf.int32)
        count_loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=true_count, logits=count_logits
            )
        )
        loss = loss + COUNT_HEAD_WEIGHT * count_loss
    return loss, xy_loss, mask_loss


def _compute_hungarian_loss(xy_true, mask_true, predictions, weights):
    pred_xy = tf.stack([predictions[2 * i] for i in range(4)], axis=1)
    pred_presence = tf.stack([predictions[2 * i + 1][:, 0] for i in range(4)], axis=1)
    w = tf.cast(weights, tf.float32)

    diff = tf.expand_dims(pred_xy, 2) - tf.expand_dims(xy_true, 1)
    l2_sq = tf.reduce_sum(tf.square(diff), axis=-1)

    mask_3d = tf.expand_dims(mask_true, 1)
    xy_cost = l2_sq * mask_3d

    gt_mask_for_bce = tf.expand_dims(mask_true, 1)
    pred_for_bce = tf.expand_dims(pred_presence, 2)
    cls_cost = (
        tf.math.maximum(pred_for_bce, 0.0)
        - pred_for_bce * gt_mask_for_bce
        + tf.math.log1p(tf.exp(-tf.abs(pred_for_bce)))
    )

    cost = xy_cost + cls_cost

    assignments = _hungarian_4x4(cost)

    matched_gt_xy = tf.gather(xy_true, assignments, batch_dims=1)
    matched_mask = tf.gather(mask_true, assignments, batch_dims=1)

    mask_exp = tf.expand_dims(matched_mask, -1)
    w_exp_3d = tf.reshape(w, [-1, 1, 1])
    sq_err = tf.square(pred_xy - matched_gt_xy) * mask_exp * w_exp_3d
    n_valid = tf.reduce_sum(matched_mask * tf.expand_dims(w, -1)) * 2.0 + 1e-8
    xy_loss = tf.reduce_sum(sq_err) / n_valid

    mask_loss = tf.constant(0.0)
    for i in range(4):
        m = matched_mask[:, i]
        mask_loss += _mask_bce(m, pred_presence[:, i], weights=w)
    mask_loss /= 4.0

    loss = xy_loss + MASK_LOSS_WEIGHT * mask_loss
    if COUNT_HEAD_WEIGHT > 0.0:
        count_logits = predictions[-1]
        true_count = tf.cast(tf.reduce_sum(mask_true, axis=1), tf.int32)
        count_loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=true_count, logits=count_logits
            )
        )
        loss = loss + COUNT_HEAD_WEIGHT * count_loss
    return loss, xy_loss, mask_loss


compute_loss = _compute_hungarian_loss if HUNGARIAN_LOSS else _compute_fixed_loss


# ==========================================================================
# Training
# ==========================================================================
@tf.function
def train_step(x, xy_true, mask_true, weights):
    with tf.GradientTape() as tape:
        predictions = model(x, training=True)
        loss, xy_l, mask_l = compute_loss(xy_true, mask_true, predictions, weights)
    grads = tape.gradient(loss, model.trainable_variables)
    grad_norm = tf.linalg.global_norm(grads)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss, xy_l, mask_l, grad_norm


def eval_epoch(dataset):
    losses = []
    for x_batch, y_batch in dataset:
        predictions = model(x_batch, training=False)
        loss, _, _ = compute_loss(
            y_batch["xy"], y_batch["mask"], predictions, y_batch["weight"]
        )
        losses.append(loss.numpy())
    return np.mean(losses)


# ==========================================================================
# Main training loop
# ==========================================================================
print(
    f"\nTraining for {EPOCHS} epochs  (lr={LEARNING_RATE}, batch={BATCH_SIZE}, schedule={LR_SCHEDULE}, warmup={WARMUP_EPOCHS})"
)
print(
    f"FFT bins kept: [{fft_bin_lo}, {fft_bin_hi}]  freq kept: [{fft_freq_lo}, {fft_freq_hi}]"
)
print("-" * 80)

best_weights = None
best_val_loss = float("inf")

for epoch in range(EPOCHS):
    train_losses, grad_norms = [], []
    for x_batch, y_batch in train_ds:
        loss, _, _, gn = train_step(
            x_batch, y_batch["xy"], y_batch["mask"], y_batch["weight"]
        )
        train_losses.append(loss.numpy())
        grad_norms.append(gn.numpy())

    val_loss = eval_epoch(val_ds)

    print(
        f"Epoch {epoch + 1:3d}/{EPOCHS}  "
        f"train_loss={np.mean(train_losses):.4f}  "
        f"val_loss={val_loss:.4f}  "
        f"grad_norm={np.mean(grad_norms):.4f}  "
        f"lr={optimizer.learning_rate.numpy() if hasattr(optimizer.learning_rate, 'numpy') else float(optimizer.learning_rate):.6f}  "
        f"best_val={best_val_loss:.4f}"
    )

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_weights = model.get_weights()

if best_weights is not None:
    model.set_weights(best_weights)
    print(f"\nRestored best model (val_loss={best_val_loss:.4f})")

# ==========================================================================
# Final evaluation: location error and people-count accuracy
# ==========================================================================


def _accumulate_metrics(
    pred_xy,
    mask_pred,
    xy_true,
    mask_true,
    xy_errs,
    count_accs,
    xy_errs_by_count,
    count_accs_by_count,
    count_confusion,
    fusion_errs_by_count,
    splitting_errs_by_count,
):
    for b in range(xy_true.shape[0]):
        true_count = int(mask_true[b].sum())

        if HUNGARIAN_LOSS:
            cost = np.zeros((4, 4))
            for i in range(4):
                for j in range(4):
                    if mask_true[b, j] > 0.5:
                        cost[i, j] = np.sum((pred_xy[b, i] - xy_true[b, j]) ** 2)
            _, col_ind = linear_sum_assignment(cost)
            for i in range(4):
                j = col_ind[i]
                if mask_true[b, j] > 0.5:
                    err = np.linalg.norm(pred_xy[b, i] - xy_true[b, j])
                    xy_errs.append(err)
                    xy_errs_by_count[true_count].append(err)
        else:
            for j in range(4):
                if mask_true[b, j] > 0.5:
                    err = np.linalg.norm(pred_xy[b, j] - xy_true[b, j])
                    xy_errs.append(err)
                    xy_errs_by_count[true_count].append(err)

        pred_count = int(mask_pred[b].sum())
        correct = true_count == pred_count
        count_accs.append(correct)
        count_accs_by_count[true_count].append(correct)
        count_confusion[true_count][pred_count] += 1

        true_xy = xy_true[b][mask_true[b] > 0.5]
        pred_xy_present = pred_xy[b][mask_pred[b]]

        if (
            pred_count == true_count - 1
            and true_count >= 1
            and pred_count >= 1
        ):
            min_dists_true = np.min(
                np.linalg.norm(
                    true_xy[:, None, :] - pred_xy_present[None, :, :], axis=-1
                ),
                axis=1,
            )
            missing_dist = float(np.max(min_dists_true))
            fusion_errs_by_count[true_count].append(missing_dist)
        elif (
            pred_count == true_count + 1
            and true_count >= 1
            and pred_count <= 4
        ):
            min_dists_pred = np.min(
                np.linalg.norm(
                    pred_xy_present[:, None, :] - true_xy[None, :, :], axis=-1
                ),
                axis=1,
            )
            fake_dist = float(np.max(min_dists_pred))
            splitting_errs_by_count[true_count].append(fake_dist)
        elif (
            pred_count == true_count + 1
            and true_count == 0
            and pred_count <= 4
        ):
            splitting_errs_by_count[true_count].append(float("nan"))


def _collect_predictions(dataset):
    pred_xy_list, pred_pr_list, xy_true_list, mask_true_list = [], [], [], []
    for x_input, y_batch in dataset:
        preds = model(x_input, training=False)
        pred_xy = np.stack([preds[2 * i].numpy() for i in range(4)], axis=1)
        pred_pr = np.stack([preds[2 * i + 1].numpy()[:, 0] for i in range(4)], axis=1)
        pred_xy_list.append(pred_xy)
        pred_pr_list.append(pred_pr)
        xy_true_list.append(y_batch["xy"].numpy())
        mask_true_list.append(y_batch["mask"].numpy())
    return (
        np.concatenate(pred_xy_list, axis=0),
        np.concatenate(pred_pr_list, axis=0),
        np.concatenate(xy_true_list, axis=0),
        np.concatenate(mask_true_list, axis=0),
    )


def apply_nms(pred_xy, pred_pr, mask_pred, nms_radius):
    if nms_radius <= 0.0:
        return mask_pred
    B = pred_xy.shape[0]
    out = np.zeros_like(mask_pred)
    order = np.argsort(-pred_pr, axis=1)
    for b in range(B):
        kept_xy = []
        for i in order[b]:
            if not mask_pred[b, i]:
                continue
            ok = True
            for kx in kept_xy:
                if np.linalg.norm(pred_xy[b, i] - kx) < nms_radius:
                    ok = False
                    break
            if ok:
                out[b, i] = True
                kept_xy.append(pred_xy[b, i])
    return out


def _predict_mask(pred_xy, pred_pr, thresholds, nms_radius):
    thresholds = np.asarray(thresholds, dtype=np.float64)
    if thresholds.ndim == 0:
        thresholds = np.full(4, float(thresholds))
    mask_pred = expit(pred_pr) > thresholds[None, :]
    return apply_nms(pred_xy, pred_pr, mask_pred, nms_radius)


def _count_accuracy(pred_xy, pred_pr, mask_true, thresholds, nms_radius):
    mask_pred = _predict_mask(pred_xy, pred_pr, thresholds, nms_radius)
    pred_count = mask_pred.sum(axis=1)
    true_count = (mask_true > 0.5).sum(axis=1)
    return float(np.mean(pred_count == true_count))


def tune_per_slot_thresholds(
    pred_xy, pred_pr, mask_true, candidates, nms_radius, iterations=2
):
    thresholds = np.full(4, 0.5)
    for _ in range(iterations):
        for slot in range(4):
            best_t, best_acc = thresholds[slot], -1.0
            for t in candidates:
                test = thresholds.copy()
                test[slot] = t
                acc = _count_accuracy(pred_xy, pred_pr, mask_true, test, nms_radius)
                if acc > best_acc:
                    best_acc, best_t = acc, t
            thresholds[slot] = best_t
    return thresholds


def tune_thresholds_and_nms(
    pred_xy, pred_pr, mask_true, threshold_candidates, nms_candidates
):
    print(f"\n=== Joint NMS + per-slot threshold tuning on validation ===")
    print(
        f"  {'nms_radius':>10s}  "
        f"{'thresholds':>28s}  {'count_acc':>10s}"
    )
    best_radius, best_thresholds, best_acc = 0.0, np.full(4, 0.5), -1.0
    for r in nms_candidates:
        thr = tune_per_slot_thresholds(
            pred_xy, pred_pr, mask_true, threshold_candidates, r
        )
        acc = _count_accuracy(pred_xy, pred_pr, mask_true, thr, r) * 100
        thr_str = "[" + ", ".join(f"{x:.2f}" for x in thr) + "]"
        print(f"  {r:>10.3f}  {thr_str:>28s}  {acc:>9.1f}%")
        if acc > best_acc:
            best_acc, best_radius, best_thresholds = acc, r, thr
    thr_str = "[" + ", ".join(f"{x:.2f}" for x in best_thresholds) + "]"
    print(
        f"  -> best: nms_radius={best_radius:.3f}  thresholds={thr_str}  "
        f"count_acc={best_acc:.1f}%"
    )
    return best_radius, best_thresholds


def evaluate_from_predictions(
    pred_xy, pred_pr, xy_true, mask_true, name, thresholds, nms_radius
):
    all_xy_err = []
    all_count_correct = []
    xy_errs_by_count = {k: [] for k in range(5)}
    count_accs_by_count = {k: [] for k in range(5)}
    count_confusion = {k: {p: 0 for p in range(5)} for k in range(5)}
    fusion_errs_by_count = {k: [] for k in range(5)}
    splitting_errs_by_count = {k: [] for k in range(5)}

    mask_pred = _predict_mask(pred_xy, pred_pr, thresholds, nms_radius)
    _accumulate_metrics(
        pred_xy,
        mask_pred,
        xy_true,
        mask_true,
        all_xy_err,
        all_count_correct,
        xy_errs_by_count,
        count_accs_by_count,
        count_confusion,
        fusion_errs_by_count,
        splitting_errs_by_count,
    )

    mean_err = np.mean(all_xy_err) if all_xy_err else float("nan")
    accuracy = np.mean(all_count_correct) * 100
    print(
        f"  {name:6s}  location_error={mean_err:.4f} m  count_accuracy={accuracy:.1f}%"
    )
    print(
        f"         {'n_people':>10s}  {'n_samples':>10s}  {'loc_err':>10s}  {'cnt_acc':>10s}"
    )
    for k in range(5):
        n = len(count_accs_by_count[k])
        e = np.mean(xy_errs_by_count[k]) if xy_errs_by_count[k] else float("nan")
        a = np.mean(count_accs_by_count[k]) * 100 if n > 0 else float("nan")
        print(f"         {k:>10d}  {n:>10d}  {e:>10.4f}  {a:>9.1f}%")

    print(f"\n         Count prediction distribution (% predicted by true count):")
    header = "         " + "true\\pred".rjust(12) + "".join(
        f"{p:>9d}" for p in range(5)
    )
    print(header)
    for k in range(5):
        total = sum(count_confusion[k].values())
        if total == 0:
            row = "         " + f"{k:>12d}" + "".join("      n/a" for _ in range(5))
        else:
            row = "         " + f"{k:>12d}" + "".join(
                f"{(count_confusion[k][p] / total * 100):>8.1f}%"
                for p in range(5)
            )
        print(row)

    print(f"\n         Off-by-one localization analysis:")
    print(
        f"         {'true_count':>10s}  "
        f"{'fusion_n':>10s}  {'fusion_err':>12s}  "
        f"{'split_n':>10s}  {'split_err':>12s}"
    )
    for k in range(5):
        fn = len(fusion_errs_by_count[k])
        fe = (
            np.nanmean(fusion_errs_by_count[k])
            if fn > 0
            else float("nan")
        )
        sn = len(splitting_errs_by_count[k])
        se = (
            np.nanmean(splitting_errs_by_count[k])
            if sn > 0
            else float("nan")
        )
        print(
            f"         {k:>10d}  "
            f"{fn:>10d}  {fe:>12.4f}  "
            f"{sn:>10d}  {se:>12.4f}"
        )

    breakdown = {}
    for k in range(5):
        n = len(count_accs_by_count[k])
        breakdown[k] = {
            "n_samples": n,
            "loc_err": np.mean(xy_errs_by_count[k])
            if xy_errs_by_count[k]
            else float("nan"),
            "cnt_acc": np.mean(count_accs_by_count[k]) * 100 if n > 0 else float("nan"),
            "count_distribution": {
                p: (count_confusion[k][p] / sum(count_confusion[k].values()) * 100)
                if sum(count_confusion[k].values()) > 0
                else float("nan")
                for p in range(5)
            },
            "fusion_err": np.nanmean(fusion_errs_by_count[k])
            if len(fusion_errs_by_count[k]) > 0
            else float("nan"),
            "splitting_err": np.nanmean(splitting_errs_by_count[k])
            if len(splitting_errs_by_count[k]) > 0
            else float("nan"),
        }
    return mean_err, accuracy, breakdown


print("\n=== Final Evaluation ===")
train_pred = _collect_predictions(train_ds)
val_pred = _collect_predictions(val_ds)
test_pred = _collect_predictions(test_ds)

THRESHOLD_CANDIDATES = [round(x, 3) for x in np.arange(0.30, 0.86, 0.05)]
NMS_CANDIDATES = [0.0, 0.3, 0.5, 0.7, 1.0] if APPLY_NMS else [0.0]
best_nms_radius, best_thresholds = tune_thresholds_and_nms(
    val_pred[0], val_pred[1], val_pred[3], THRESHOLD_CANDIDATES, NMS_CANDIDATES
)

print(f"\n--- Evaluation @ default threshold=0.500, nms=0.0 ---")
evaluate_from_predictions(*train_pred, "Train", 0.5, 0.0)
evaluate_from_predictions(*val_pred, "Val", 0.5, 0.0)
evaluate_from_predictions(*test_pred, "Test", 0.5, 0.0)

thr_str = "[" + ", ".join(f"{x:.2f}" for x in best_thresholds) + "]"
print(
    f"\n--- Evaluation @ tuned thresholds={thr_str}, nms_radius={best_nms_radius:.3f} ---"
)
train_err, train_acc, train_bk = evaluate_from_predictions(
    *train_pred, "Train", best_thresholds, best_nms_radius
)
val_err, val_acc, val_bk = evaluate_from_predictions(
    *val_pred, "Val", best_thresholds, best_nms_radius
)
test_err, test_acc, test_bk = evaluate_from_predictions(
    *test_pred, "Test", best_thresholds, best_nms_radius
)

# ==========================================================================
# Log results to CSV
# ==========================================================================
hp_str = (
    f"SEED={SEED} SUB_WINDOW_SIZE={SUB_WINDOW_SIZE} "
    f"TRAIN_RATIO={TRAIN_RATIO} VAL_RATIO={VAL_RATIO} "
    f"BATCH_SIZE={BATCH_SIZE} EPOCHS={EPOCHS} "
    f"LEARNING_RATE={LEARNING_RATE} STD_EPSILON={STD_EPSILON} "
    f"BACKGROUND_MODE={BACKGROUND_MODE} EMA_ALPHA={EMA_ALPHA} "
    f"WEIGHT_DECAY={WEIGHT_DECAY} DROPOUT={DROPOUT} "
    f"HUNGARIAN_LOSS={HUNGARIAN_LOSS} "
    f"LR_SCHEDULE={LR_SCHEDULE} WARMUP_EPOCHS={WARMUP_EPOCHS} "
    f"MODEL=attn_ant_fft D_ATTN={D_ATTN} N_HEADS={N_HEADS} "
    f"CONV_FILTERS={CONV_FILTERS} CONV_KERNELS={CONV_KERNELS} "
    f"DENSE_DIM={DENSE_DIM} HEAD_DIM={HEAD_DIM} POS_EMBED={POS_EMBED} "
    f"ANTENNA_D_PROJ={ANTENNA_D_PROJ} "
    f"ENCODER_POOLING={ENCODER_POOLING} ENCODER_TOPK={ENCODER_TOPK} "
    f"ENCODER_ATTN_TEMPERATURE={ENCODER_ATTN_TEMPERATURE} "
    f"FFT_BIN_RANGE={FFT_BIN_RANGE} FFT_FREQ_RANGE={FFT_FREQ_RANGE} "
    f"TRANS_BLOCKS={TRANS_BLOCKS} FFN_MULT={FFN_MULT} "
    f"MACRO_WINDOW_SPLIT={MACRO_WINDOW_SPLIT} "
    f"USE_FOCAL_LOSS={USE_FOCAL_LOSS} AMPLITUDE_CHANNEL={AMPLITUDE_CHANNEL} "
    f"ANTENNA_DROPOUT={ANTENNA_DROPOUT} PHASE_SHIFT_MAX={PHASE_SHIFT_MAX} "
    f"MASK_LOSS_WEIGHT={MASK_LOSS_WEIGHT} "
    f"CLASS_BALANCE={CLASS_BALANCE} CLASS_LOSS_SCALE={CLASS_LOSS_SCALE} "
    f"APPLY_NMS={APPLY_NMS} COUNT_HEAD_WEIGHT={COUNT_HEAD_WEIGHT}"
)

csv_path = os.environ.get("RESULTS_CSV", os.path.join(os.path.dirname(__file__) or ".", "attn_ant_fft_results.csv"))
n_params = model.count_params()
header = [
    "experiment_name",
    "n_params",
    "train_location_error",
    "train_count_accuracy",
    "val_location_error",
    "val_count_accuracy",
    "test_location_error",
    "test_count_accuracy",
    "hyperparameters",
]
row = [
    EXPERIMENT_NAME,
    n_params,
    f"{train_err:.4f}",
    f"{train_acc:.1f}",
    f"{val_err:.4f}",
    f"{val_acc:.1f}",
    f"{test_err:.4f}",
    f"{test_acc:.1f}",
    hp_str,
]

write_header = not os.path.exists(csv_path)
with open(csv_path, "a", newline="") as f:
    writer = csv.writer(f)
    if write_header:
        writer.writerow(header)
    writer.writerow(row)

print(f"\nResults appended to {csv_path}")

# ==========================================================================
# Export model
# ==========================================================================
os.makedirs(model_dir, exist_ok=True)
model.save(os.path.join(model_dir, "model.keras"))
print(f"Model exported to {model_dir}/model.keras")

# ==========================================================================
# INT8 Post-Training Quantization & TFLite export
# ==========================================================================


def _representative_dataset():
    n_samples = min(500, len(train_features))
    for i in range(n_samples):
        yield [train_features[i : i + 1].astype(np.float32)]


converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = _representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

tflite_model = converter.convert()

interpreter = tf.lite.Interpreter(model_content=tflite_model)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = sorted(interpreter.get_output_details(), key=lambda o: o["name"])

input_scale, input_zero_point = input_details[0]["quantization"]

xy_out = [o for o in output_details if "xy" in o["name"]]
pr_out = [o for o in output_details if "presence" in o["name"]]

if len(xy_out) != 4 or len(pr_out) != 4:
    xy_out = sorted(
        [o for o in output_details if np.prod(o["shape"]) == 2], key=lambda o: o["name"]
    )
    pr_out = sorted(
        [o for o in output_details if np.prod(o["shape"]) == 1], key=lambda o: o["name"]
    )

keras_xy_err, tflite_xy_err = [], []
keras_count_ok, tflite_count_ok = [], []

for idx in range(len(test_features)):
    sample = test_features[idx : idx + 1]
    mask_true = test_mask[idx]
    xy_true = test_xy[idx]

    keras_out = model(sample, training=False)
    keras_xy = np.stack([keras_out[2 * i].numpy()[0] for i in range(4)], axis=0)
    keras_pr = np.stack([keras_out[2 * i + 1].numpy()[0, 0] for i in range(4)], axis=0)
    keras_mask_pred = _predict_mask(
        keras_xy[None, ...], keras_pr[None, ...], best_thresholds, best_nms_radius
    )[0]

    quantized = np.round(sample / input_scale + input_zero_point).astype(np.int8)
    interpreter.set_tensor(input_details[0]["index"], quantized)
    interpreter.invoke()

    tflite_xy_all, tflite_pr_all = [], []
    for o in xy_out:
        raw = interpreter.get_tensor(o["index"]).flatten()
        scale, zp = o["quantization"]
        tflite_xy_all.append((raw.astype(np.float32) - zp) * scale)
    for o in pr_out:
        raw = interpreter.get_tensor(o["index"]).flatten()[0]
        scale, zp = o["quantization"]
        tflite_pr_all.append((float(raw) - zp) * scale)
    tflite_xy = np.stack(tflite_xy_all, axis=0)
    tflite_pr_arr = np.array(tflite_pr_all)
    tflite_mask_pred = _predict_mask(
        tflite_xy[None, ...], tflite_pr_arr[None, ...], best_thresholds, best_nms_radius
    )[0]

    if HUNGARIAN_LOSS:
        cost = np.zeros((4, 4))
        for pi in range(4):
            for pj in range(4):
                if mask_true[pj] > 0.5:
                    cost[pi, pj] = np.sum((keras_xy[pi] - xy_true[pj]) ** 2)
        _, col_ind = linear_sum_assignment(cost)
        for pi, pj in enumerate(col_ind):
            if mask_true[pj] > 0.5:
                keras_xy_err.append(np.linalg.norm(keras_xy[pi] - xy_true[pj]))
                tflite_xy_err.append(np.linalg.norm(tflite_xy[pi] - xy_true[pj]))
    else:
        for j in range(4):
            if mask_true[j] > 0.5:
                keras_xy_err.append(np.linalg.norm(keras_xy[j] - xy_true[j]))
                tflite_xy_err.append(np.linalg.norm(tflite_xy[j] - xy_true[j]))

    true_count = int(mask_true.sum())
    keras_count_ok.append(int(keras_mask_pred.sum()) == true_count)
    tflite_count_ok.append(int(tflite_mask_pred.sum()) == true_count)

keras_loc_err = np.mean(keras_xy_err) if keras_xy_err else float("nan")
tflite_loc_err = np.mean(tflite_xy_err) if tflite_xy_err else float("nan")
keras_cnt_acc = np.mean(keras_count_ok) * 100
tflite_cnt_acc = np.mean(tflite_count_ok) * 100
loc_degradation = (
    (tflite_loc_err - keras_loc_err) / keras_loc_err * 100
    if keras_xy_err
    else float("nan")
)
cnt_degradation = tflite_cnt_acc - keras_cnt_acc

tflite_size_kb = len(tflite_model) / 1024

print("\n=== TFLite INT8 verification (full test set) ===")
print(f"  samples verified: {len(test_features)}")
print(f"  Keras  location_error = {keras_loc_err:.4f} m")
print(f"  TFLite location_error = {tflite_loc_err:.4f} m")
print(f"  degradation           = {loc_degradation:+.1f}%")
print(f"  Keras  count_accuracy = {keras_cnt_acc:.1f}%")
print(f"  TFLite count_accuracy = {tflite_cnt_acc:.1f}%")
print(f"  TFLite model size     = {tflite_size_kb:.1f} KB")

tflite_path = os.path.join(model_dir, "model_int8.tflite")
with open(tflite_path, "wb") as f:
    f.write(tflite_model)
print(f"INT8 TFLite model saved to {tflite_path}")

# ==========================================================================
# Export to submission directory
# ==========================================================================
submission_dir = os.path.join(os.path.dirname(__file__) or ".", "submission")
os.makedirs(submission_dir, exist_ok=True)

import shutil

shutil.copy2(tflite_path, os.path.join(submission_dir, "model.tflite"))
print(f"TFLite model copied to {submission_dir}/model.tflite")

np.savez(
    os.path.join(submission_dir, "model_params"),
    mean=feat_mean,
    std=feat_std,
    baseline=bl_med if BACKGROUND_MODE == "baseline" else np.zeros_like(feat_mean),
    has_baseline=np.array(BACKGROUND_MODE == "baseline"),
    fft_bin_lo=fft_bin_lo,
    fft_bin_hi=fft_bin_hi,
    fft_freq_lo=fft_freq_lo,
    fft_freq_hi=fft_freq_hi,
    sub_window_size=SUB_WINDOW_SIZE,
    std_epsilon=STD_EPSILON,
    thresholds=best_thresholds,
    nms_radius=best_nms_radius,
    background_mode=BACKGROUND_MODE,
    ema_alpha=EMA_ALPHA,
    amplitude_channel=np.array(AMPLITUDE_CHANNEL),
)
print(f"Model params saved to {submission_dir}/model_params.npz")

# Also save to volume for retrieval
shutil.copy2(
    os.path.join(submission_dir, "model_params.npz"),
    os.path.join(model_dir, "model_params.npz")
)
print(f"Model params also saved to {model_dir}/model_params.npz")

# ==========================================================================
# Log quantization results to CSV
# ==========================================================================
q_csv_path = os.environ.get(
    "QUANT_RESULTS_CSV", os.path.join(os.path.dirname(__file__) or ".", "attn_ant_fft_quant_results.csv")
)
q_header = [
    "experiment_name",
    "qt_type",
    "n_params",
    "tflite_size_kb",
    "keras_location_error",
    "keras_count_accuracy",
    "tflite_location_error",
    "tflite_count_accuracy",
    "location_degradation_pct",
    "count_accuracy_degradation_pct",
    "hyperparameters",
]
q_row = [
    EXPERIMENT_NAME,
    "PTQ",
    n_params,
    f"{tflite_size_kb:.1f}",
    f"{keras_loc_err:.4f}",
    f"{keras_cnt_acc:.1f}",
    f"{tflite_loc_err:.4f}",
    f"{tflite_cnt_acc:.1f}",
    f"{loc_degradation:+.1f}",
    f"{cnt_degradation:+.1f}",
    hp_str,
]

write_header = not os.path.exists(q_csv_path)
with open(q_csv_path, "a", newline="") as f:
    writer = csv.writer(f)
    if write_header:
        writer.writerow(q_header)
    writer.writerow(q_row)

print(f"PTQ results appended to {q_csv_path}")
