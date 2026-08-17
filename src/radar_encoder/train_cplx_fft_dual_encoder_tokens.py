import os
import glob
import csv
import json
import itertools
import re
import numpy as np
import tensorflow as tf
from scipy.optimize import linear_sum_assignment
from scipy.special import expit

physical_gpus = tf.config.list_physical_devices("GPU")
print("TensorFlow GPU devices:")
if physical_gpus:
    for idx, gpu in enumerate(physical_gpus):
        print(f"  GPU:{idx} physical={gpu.name}")
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as exc:
            print(f"  GPU:{idx} memory_growth not changed: {exc}")
else:
    print("  none")
logical_gpus = tf.config.list_logical_devices("GPU")
if logical_gpus:
    for idx, gpu in enumerate(logical_gpus):
        print(f"  GPU:{idx} logical={gpu.name}")

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
EXPERIMENT_NAME = os.environ.get("EXPERIMENT_NAME", "default")
CONV_FILTERS = os.environ.get("CONV_FILTERS", "32,48")
COMPLEX_CONV_FILTERS = os.environ.get("COMPLEX_CONV_FILTERS", CONV_FILTERS)
AMPLITUDE_CONV_FILTERS = os.environ.get("AMPLITUDE_CONV_FILTERS", CONV_FILTERS)
CONV_KERNELS = os.environ.get("CONV_KERNELS", "5,3")
CONV_BLOCK_TYPE = os.environ.get("CONV_BLOCK_TYPE", "conv2d").lower()
MOBILENETV2_EXPANSION = float(os.environ.get("MOBILENETV2_EXPANSION", 2.0))
D_ATTN = int(os.environ.get("D_ATTN", 16))
N_HEADS = int(os.environ.get("N_HEADS", 1))
DENSE_DIM = int(os.environ.get("DENSE_DIM", 64))
HEAD_DIM = int(os.environ.get("HEAD_DIM", 16))
POS_EMBED = bool(int(os.environ.get("POS_EMBED", 0)))
FFT_BIN_RANGE = os.environ.get("FFT_BIN_RANGE", "0,119")
FFT_FREQ_RANGE = os.environ.get("FFT_FREQ_RANGE", "0,25")
LR_SCHEDULE = os.environ.get("LR_SCHEDULE", "none")
WARMUP_EPOCHS = int(os.environ.get("WARMUP_EPOCHS", 5))
TRANS_BLOCKS = int(os.environ.get("TRANS_BLOCKS", 1))
FFN_MULT = float(os.environ.get("FFN_MULT", 4))
NORM_TYPE = str(os.environ.get("NORM_TYPE", "batch"))
ENCODER_POOLING = os.environ.get("ENCODER_POOLING", "gap")
COMPLEX_ENCODER_POOLING = os.environ.get("COMPLEX_ENCODER_POOLING", ENCODER_POOLING)
AMPLITUDE_ENCODER_POOLING = os.environ.get(
    "AMPLITUDE_ENCODER_POOLING", ENCODER_POOLING
)
ENCODER_TOPK = int(os.environ.get("ENCODER_TOPK", 8))
ENCODER_ATTN_TEMPERATURE = float(os.environ.get("ENCODER_ATTN_TEMPERATURE", 4.0))
COUNT_LOSS_WEIGHT = float(os.environ.get("COUNT_LOSS_WEIGHT", 0.0))
LOCATION_LOSS = os.environ.get("LOCATION_LOSS", "mse").lower()
LOCATION_HUBER_DELTA = float(os.environ.get("LOCATION_HUBER_DELTA", 0.5))
LOCATION_CHARBONNIER_EPS = float(os.environ.get("LOCATION_CHARBONNIER_EPS", 0.5))
AMPLITUDE_CHANNEL = True
ANTENNA_DROPOUT = float(os.environ.get("ANTENNA_DROPOUT", 0.0))
PHASE_SHIFT_MAX = float(os.environ.get("PHASE_SHIFT_MAX", 0.0))
MASK_LOSS_WEIGHT = float(os.environ.get("MASK_LOSS_WEIGHT", 1.0))
CLASS_BALANCE = bool(int(os.environ.get("CLASS_BALANCE", 0)))
CLASS_LOSS_SCALE = bool(int(os.environ.get("CLASS_LOSS_SCALE", 0)))
TOPK_PTQ_SELECTION = bool(int(os.environ.get("TOPK_PTQ_SELECTION", 0)))
TOPK_PTQ_K = int(os.environ.get("TOPK_PTQ_K", 5))
TOPK_PTQ_EVAL_EVERY = int(os.environ.get("TOPK_PTQ_EVAL_EVERY", 1))
TOPK_SELECTION_COUNT_WEIGHT = float(
    os.environ.get("TOPK_SELECTION_COUNT_WEIGHT", 0.002)
)
QAT = bool(int(os.environ.get("QAT", 0)))
QAT_EPOCHS = int(os.environ.get("QAT_EPOCHS", 10))
QAT_LEARNING_RATE = float(os.environ.get("QAT_LEARNING_RATE", LEARNING_RATE * 0.1))
TOPK_PTQ_K = max(1, TOPK_PTQ_K)
TOPK_PTQ_EVAL_EVERY = max(1, TOPK_PTQ_EVAL_EVERY)

fft_bin_lo, fft_bin_hi = [int(x) for x in FFT_BIN_RANGE.split(",")]
fft_freq_lo, fft_freq_hi = [int(x) for x in FFT_FREQ_RANGE.split(",")]

# ==========================================================================
# Reproducibility
# ==========================================================================
np.random.seed(SEED)
tf.random.set_seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)

# ==========================================================================
# Model definition: per-radar Conv2D (on complex FFT spectrogram) + attention + 4 heads
# ==========================================================================
N_RADARS = 6
N_ANTENNAS = 3
N_BINS = 120
N_FREQ = SUB_WINDOW_SIZE

n_bins_kept = fft_bin_hi - fft_bin_lo + 1
n_freq_kept = fft_freq_hi - fft_freq_lo + 1
n_channels = 3
n_attention_tokens = N_RADARS * 2

complex_conv_filters = [int(x) for x in COMPLEX_CONV_FILTERS.split(",")]
amplitude_conv_filters = [int(x) for x in AMPLITUDE_CONV_FILTERS.split(",")]
conv_kernels = [int(x) for x in CONV_KERNELS.split(",")]
assert len(complex_conv_filters) == len(conv_kernels), (
    "COMPLEX_CONV_FILTERS and CONV_KERNELS must have same length"
)
assert len(amplitude_conv_filters) == len(conv_kernels), (
    "AMPLITUDE_CONV_FILTERS and CONV_KERNELS must have same length"
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
SUPPORTED_CONV_BLOCK_TYPES = (
    "conv2d",
    "depthwise_separable",
    "separable_conv2d",
    "mobilenetv2",
)
if CONV_BLOCK_TYPE not in SUPPORTED_CONV_BLOCK_TYPES:
    raise ValueError(
        f"Unsupported CONV_BLOCK_TYPE: {CONV_BLOCK_TYPE}. "
        f"Supported values: {SUPPORTED_CONV_BLOCK_TYPES}"
    )
if MOBILENETV2_EXPANSION <= 0.0:
    raise ValueError(
        f"MOBILENETV2_EXPANSION must be > 0, got: {MOBILENETV2_EXPANSION}"
    )
SUPPORTED_ENCODER_POOLINGS = (
    "gap",
    "topk",
    "topk8",
    "topk16",
    "gap_topk",
    "gap_topk8",
    "gap_soft_attn",
    "gap_soft_attn_temp4",
    "soft_attn",
    "soft_attn_temp2",
    "soft_attn_temp4",
    "soft_attn_temp6",
    "soft_attn_temp8",
    "topk_soft_attn",
    "topk8_soft_attn",
    "topk8_soft_attn_temp4",
)
for pooling_name, pooling_value in (
    ("ENCODER_POOLING", ENCODER_POOLING),
    ("COMPLEX_ENCODER_POOLING", COMPLEX_ENCODER_POOLING),
    ("AMPLITUDE_ENCODER_POOLING", AMPLITUDE_ENCODER_POOLING),
):
    if pooling_value not in SUPPORTED_ENCODER_POOLINGS:
        raise ValueError(f"Unsupported {pooling_name}: {pooling_value}")
if ENCODER_TOPK <= 0:
    raise ValueError(f"ENCODER_TOPK must be > 0, got: {ENCODER_TOPK}")
if ENCODER_ATTN_TEMPERATURE <= 0.0:
    raise ValueError(
        f"ENCODER_ATTN_TEMPERATURE must be > 0, got: {ENCODER_ATTN_TEMPERATURE}"
    )
if COUNT_LOSS_WEIGHT < 0.0:
    raise ValueError(f"COUNT_LOSS_WEIGHT must be >= 0, got: {COUNT_LOSS_WEIGHT}")
if LOCATION_LOSS not in ("mse", "huber", "charbonnier"):
    raise ValueError(f"Unsupported LOCATION_LOSS: {LOCATION_LOSS}")
if LOCATION_HUBER_DELTA <= 0.0:
    raise ValueError(
        f"LOCATION_HUBER_DELTA must be > 0, got: {LOCATION_HUBER_DELTA}"
    )
if LOCATION_CHARBONNIER_EPS <= 0.0:
    raise ValueError(
        f"LOCATION_CHARBONNIER_EPS must be > 0, got: {LOCATION_CHARBONNIER_EPS}"
    )
if QAT_EPOCHS < 0:
    raise ValueError(f"QAT_EPOCHS must be >= 0, got: {QAT_EPOCHS}")
if QAT_LEARNING_RATE <= 0.0:
    raise ValueError(f"QAT_LEARNING_RATE must be > 0, got: {QAT_LEARNING_RATE}")

tfmot = None
if QAT:
    try:
        import tensorflow_model_optimization as tfmot
    except ImportError as exc:
        raise RuntimeError(
            "QAT=1 requires tensorflow_model_optimization in the active environment"
        ) from exc


def apply_ema_iq(iq_data, alpha):
    iq_f = iq_data.astype(np.float64)
    bg = iq_f[0].copy()
    out = np.empty_like(iq_f)
    for t in range(iq_f.shape[0]):
        out[t] = iq_f[t] - bg
        bg = alpha * iq_f[t] + (1.0 - alpha) * bg
    return out.astype(iq_data.dtype)


def _mask_bce(labels, logits, weights=None):
    per_sample = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits)
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
        amp = inputs[..., 2:3]
        return tf.concat([new_real, new_imag, amp], axis=-1)


@tf.keras.utils.register_keras_serializable()
class RadarBatchFlatten(tf.keras.layers.Layer):
    def __init__(self, n_bins, n_freq, n_channels_flat, **kwargs):
        super().__init__(**kwargs)
        self.n_bins = n_bins
        self.n_freq = n_freq
        self.n_channels_flat = n_channels_flat

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "n_bins": self.n_bins,
                "n_freq": self.n_freq,
                "n_channels_flat": self.n_channels_flat,
            }
        )
        return config

    def call(self, inputs):
        shape = tf.shape(inputs)
        return tf.reshape(
            inputs,
            [shape[0] * shape[1], self.n_bins, self.n_freq, self.n_channels_flat],
        )


@tf.keras.utils.register_keras_serializable()
class ComplexFFTChannelSplit(tf.keras.layers.Layer):
    def call(self, inputs):
        return tf.concat([inputs[..., 0:2], inputs[..., 3:5], inputs[..., 6:8]], axis=-1)


@tf.keras.utils.register_keras_serializable()
class AmplitudeFFTChannelSplit(tf.keras.layers.Layer):
    def call(self, inputs):
        return tf.concat([inputs[..., 2:3], inputs[..., 5:6], inputs[..., 8:9]], axis=-1)


@tf.keras.utils.register_keras_serializable()
class RadarTokenReshape(tf.keras.layers.Layer):
    def __init__(self, n_radars, pool_dim, **kwargs):
        super().__init__(**kwargs)
        self.n_radars = n_radars
        self.pool_dim = pool_dim

    def get_config(self):
        config = super().get_config()
        config.update({"n_radars": self.n_radars, "pool_dim": self.pool_dim})
        return config

    def call(self, inputs):
        return tf.reshape(inputs, [-1, self.n_radars, self.pool_dim])


@tf.keras.utils.register_keras_serializable()
class RadarPositionEmbedding(tf.keras.layers.Layer):
    def __init__(self, n_tokens, d_model, **kwargs):
        super().__init__(**kwargs)
        self.n_tokens = n_tokens
        self.d_model = d_model

    def build(self, input_shape):
        self.pos_embed = self.add_weight(
            name="radar_pos_embed",
            shape=(1, self.n_tokens, self.d_model),
            initializer=tf.keras.initializers.RandomNormal(stddev=0.02),
            trainable=True,
        )

    def get_config(self):
        config = super().get_config()
        config.update({"n_tokens": self.n_tokens, "d_model": self.d_model})
        return config

    def call(self, inputs):
        return inputs + self.pos_embed


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


def norm_relu(x, relu6=False):
    x = norm()(x)
    if relu6:
        return tf.keras.layers.ReLU(max_value=6.0)(x)
    return tf.keras.layers.ReLU()(x)


def conv_encoder_block(x, filters, kernel_size, prefix, block_idx):
    if CONV_BLOCK_TYPE == "conv2d":
        x = tf.keras.layers.Conv2D(
            filters,
            kernel_size=(kernel_size, kernel_size),
            padding="same",
            name=f"{prefix}_conv2d_{block_idx}",
        )(x)
        x = norm_relu(x)
    elif CONV_BLOCK_TYPE == "depthwise_separable":
        x = tf.keras.layers.DepthwiseConv2D(
            kernel_size=(kernel_size, kernel_size),
            padding="same",
            name=f"{prefix}_dwconv2d_{block_idx}",
        )(x)
        x = norm_relu(x)
        x = tf.keras.layers.Conv2D(
            filters,
            kernel_size=(1, 1),
            padding="same",
            name=f"{prefix}_pwconv2d_{block_idx}",
        )(x)
        x = norm_relu(x)
    elif CONV_BLOCK_TYPE == "separable_conv2d":
        x = tf.keras.layers.SeparableConv2D(
            filters,
            kernel_size=(kernel_size, kernel_size),
            padding="same",
            name=f"{prefix}_sepconv2d_{block_idx}",
        )(x)
        x = norm_relu(x)
    elif CONV_BLOCK_TYPE == "mobilenetv2":
        residual = x
        input_channels = x.shape[-1]
        if input_channels is None:
            expanded_channels = max(1, int(round(filters * MOBILENETV2_EXPANSION)))
            use_residual = False
        else:
            input_channels = int(input_channels)
            expanded_channels = max(
                1, int(round(input_channels * MOBILENETV2_EXPANSION))
            )
            use_residual = input_channels == filters

        x = tf.keras.layers.Conv2D(
            expanded_channels,
            kernel_size=(1, 1),
            padding="same",
            name=f"{prefix}_mbv2_expand_{block_idx}",
        )(x)
        x = norm_relu(x, relu6=True)
        x = tf.keras.layers.DepthwiseConv2D(
            kernel_size=(kernel_size, kernel_size),
            padding="same",
            name=f"{prefix}_mbv2_dwconv2d_{block_idx}",
        )(x)
        x = norm_relu(x, relu6=True)
        x = tf.keras.layers.Conv2D(
            filters,
            kernel_size=(1, 1),
            padding="same",
            name=f"{prefix}_mbv2_project_{block_idx}",
        )(x)
        x = norm()(x)
        if use_residual:
            x = tf.keras.layers.Add(name=f"{prefix}_mbv2_residual_{block_idx}")(
                [x, residual]
            )
    else:
        raise ValueError(f"Unsupported CONV_BLOCK_TYPE: {CONV_BLOCK_TYPE}")

    return tf.keras.layers.MaxPool2D(pool_size=(2, 2))(x)


@tf.keras.utils.register_keras_serializable()
class TopKAveragePooling2D(tf.keras.layers.Layer):
    def __init__(self, k=8, **kwargs):
        super().__init__(**kwargs)
        self.k = k

    def get_config(self):
        config = super().get_config()
        config.update({"k": self.k})
        return config

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])

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

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])

    def call(self, inputs):
        shape = tf.shape(inputs)
        scores = self.score_proj(inputs) / self.temperature
        scores = tf.reshape(scores, [shape[0], -1, 1])
        weights = tf.nn.softmax(scores, axis=1)
        flat = tf.reshape(inputs, [shape[0], -1, shape[-1]])
        return tf.reduce_sum(flat * weights, axis=1)


def _topk_for_pooling(pooling):
    if pooling in ("topk8", "gap_topk8", "topk8_soft_attn", "topk8_soft_attn_temp4"):
        return 8
    if pooling == "topk16":
        return 16
    return ENCODER_TOPK


def _attention_temperature_for_pooling(pooling):
    if pooling in (
        "soft_attn_temp2",
    ):
        return 2.0
    if pooling in (
        "soft_attn_temp4",
        "gap_soft_attn_temp4",
        "topk8_soft_attn_temp4",
    ):
        return 4.0
    if pooling == "soft_attn_temp6":
        return 6.0
    if pooling == "soft_attn_temp8":
        return 8.0
    return ENCODER_ATTN_TEMPERATURE


def _pool_dim(pooling, conv_dim):
    if pooling in (
        "gap_topk",
        "gap_topk8",
        "gap_soft_attn",
        "gap_soft_attn_temp4",
        "topk_soft_attn",
        "topk8_soft_attn",
        "topk8_soft_attn_temp4",
    ):
        return conv_dim * 2
    return conv_dim


def encoder_pool(x, name, pooling):
    if pooling == "gap":
        return tf.keras.layers.GlobalAveragePooling2D(name=f"{name}_gap")(x)
    if pooling in ("topk", "topk8", "topk16"):
        return TopKAveragePooling2D(_topk_for_pooling(pooling), name=f"{name}_topk")(x)
    if pooling in ("gap_topk", "gap_topk8"):
        gap = tf.keras.layers.GlobalAveragePooling2D(name=f"{name}_gap")(x)
        topk = TopKAveragePooling2D(_topk_for_pooling(pooling), name=f"{name}_topk")(x)
        return tf.keras.layers.Concatenate(name=f"{name}_gap_topk")([gap, topk])
    if pooling in (
        "soft_attn",
        "soft_attn_temp2",
        "soft_attn_temp4",
        "soft_attn_temp6",
        "soft_attn_temp8",
    ):
        temperature = _attention_temperature_for_pooling(pooling)
        return SoftAttentionPooling2D(temperature, name=f"{name}_soft_attn")(x)
    if pooling in ("gap_soft_attn", "gap_soft_attn_temp4"):
        gap = tf.keras.layers.GlobalAveragePooling2D(name=f"{name}_gap")(x)
        temperature = _attention_temperature_for_pooling(pooling)
        attn = SoftAttentionPooling2D(temperature, name=f"{name}_soft_attn")(x)
        return tf.keras.layers.Concatenate(name=f"{name}_gap_soft_attn")([gap, attn])
    if pooling in ("topk_soft_attn", "topk8_soft_attn", "topk8_soft_attn_temp4"):
        topk = TopKAveragePooling2D(_topk_for_pooling(pooling), name=f"{name}_topk")(x)
        temperature = _attention_temperature_for_pooling(pooling)
        attn = SoftAttentionPooling2D(temperature, name=f"{name}_soft_attn")(x)
        return tf.keras.layers.Concatenate(name=f"{name}_topk_soft_attn")([topk, attn])
    raise ValueError(f"Unsupported encoder pooling: {pooling}")


inp = tf.keras.Input(
    shape=(N_RADARS, N_ANTENNAS, n_bins_kept, n_freq_kept, n_channels), name="input"
)

x = inp
if PHASE_SHIFT_MAX > 0.0:
    x = RandomPhaseShift(PHASE_SHIFT_MAX)(x)
if ANTENNA_DROPOUT > 0.0:
    x = AntennaDropout(ANTENNA_DROPOUT)(x)

x = tf.keras.layers.Permute((1, 3, 4, 2, 5))(x)
if QAT:
    x = RadarBatchFlatten(n_bins_kept, n_freq_kept, N_ANTENNAS * n_channels)(x)
else:
    x = tf.keras.layers.Lambda(
        lambda t: tf.reshape(
            t, [-1, n_bins_kept, n_freq_kept, N_ANTENNAS * n_channels]
        )
    )(x)

if QAT:
    x_cplx = ComplexFFTChannelSplit(name="complex_fft_channels")(x)
    x_amp = AmplitudeFFTChannelSplit(name="amplitude_fft_channel")(x)
else:
    x_cplx = tf.keras.layers.Lambda(
        lambda t: tf.concat([t[..., 0:2], t[..., 3:5], t[..., 6:8]], axis=-1),
        name="complex_fft_channels",
    )(x)
    x_amp = tf.keras.layers.Lambda(
        lambda t: tf.concat([t[..., 2:3], t[..., 5:6], t[..., 8:9]], axis=-1),
        name="amplitude_fft_channel",
    )(x)

for i, (cplx_f, amp_f, k) in enumerate(
    zip(complex_conv_filters, amplitude_conv_filters, conv_kernels)
):
    x_cplx = conv_encoder_block(x_cplx, cplx_f, k, "complex", i)
    x_amp = conv_encoder_block(x_amp, amp_f, k, "amplitude", i)

complex_last_conv_dim = complex_conv_filters[-1]
amplitude_last_conv_dim = amplitude_conv_filters[-1]
complex_pool_dim = _pool_dim(COMPLEX_ENCODER_POOLING, complex_last_conv_dim)
amplitude_pool_dim = _pool_dim(AMPLITUDE_ENCODER_POOLING, amplitude_last_conv_dim)

x_cplx = encoder_pool(x_cplx, "complex", COMPLEX_ENCODER_POOLING)
if QAT:
    x_cplx = RadarTokenReshape(N_RADARS, complex_pool_dim)(x_cplx)
else:
    x_cplx = tf.keras.layers.Lambda(
        lambda t: tf.reshape(t, [-1, N_RADARS, complex_pool_dim])
    )(x_cplx)
x_amp = encoder_pool(x_amp, "amplitude", AMPLITUDE_ENCODER_POOLING)
if QAT:
    x_amp = RadarTokenReshape(N_RADARS, amplitude_pool_dim)(x_amp)
else:
    x_amp = tf.keras.layers.Lambda(
        lambda t: tf.reshape(t, [-1, N_RADARS, amplitude_pool_dim])
    )(x_amp)

d_model = D_ATTN * N_HEADS

x_cplx = tf.keras.layers.Dense(d_model, name="complex_token_projection")(x_cplx)
x_amp = tf.keras.layers.Dense(d_model, name="amplitude_token_projection")(x_amp)
x = tf.keras.layers.Concatenate(axis=1)([x_cplx, x_amp])

if POS_EMBED:
    if QAT:
        x = RadarPositionEmbedding(n_attention_tokens, d_model)(x)
    else:
        radar_pe = tf.Variable(
            tf.random.normal([1, n_attention_tokens, d_model], stddev=0.02),
            trainable=True,
            name="radar_pos_embed",
        )
        x = tf.keras.layers.Lambda(lambda t: t + radar_pe)(x)

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

x = tf.keras.layers.Reshape((n_attention_tokens * d_model,))(x)

x = tf.keras.layers.Dense(DENSE_DIM)(x)
x = norm()(x)
x = tf.keras.layers.ReLU()(x)
x = tf.keras.layers.Dropout(DROPOUT)(x)
shared_features = x

outputs = []
for i in range(4):
    hi = tf.keras.layers.Dense(HEAD_DIM, activation="relu")(shared_features)
    xy_i = tf.keras.layers.Dense(2, name=f"p{i}_xy")(hi)
    presence_i = tf.keras.layers.Dense(1, name=f"p{i}_presence")(hi)
    outputs.extend([xy_i, presence_i])
if COUNT_LOSS_WEIGHT > 0.0:
    count_hi = tf.keras.layers.Dense(HEAD_DIM, activation="relu")(shared_features)
    count_logits = tf.keras.layers.Dense(5, name="count_logits")(count_hi)
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
        target_xy = xy_m[:, -1, :, :].astype(np.float32)
        target_mask = mask_m[:, -1, :].astype(np.float32)

        macro_features.append(features)
        macro_target_xy.append(target_xy)
        macro_target_mask.append(target_mask)
        file_basename = os.path.basename(fpath)
        file_idx = int(file_basename.replace("window_", "").replace(".npz", ""))
        mw_start = m_idx * macro_size * chunk_size
        mw_n_frames = macro_size * chunk_size
        mw_end = mw_start + mw_n_frames - 1
        macro_sources.append(
            {
                "file_path": fpath,
                "file_idx": file_idx,
                "macro_idx": m_idx,
                "start_frame": mw_start,
                "end_frame": mw_end,
                "n_frames": mw_n_frames,
            }
        )

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
    class_weights_lookup[nonzero] = len(train_counts) / (
        n_classes * class_counts[nonzero]
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
baseline = None
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
        baseline = bl_med
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


def _location_loss(pred_xy, matched_gt_xy, matched_mask, weights):
    diff = pred_xy - matched_gt_xy
    mask_exp = tf.expand_dims(matched_mask, -1)
    w_exp = tf.reshape(weights, [-1, 1, 1])

    if LOCATION_LOSS == "mse":
        sq_err = tf.square(diff) * mask_exp * w_exp
        n_valid = tf.reduce_sum(matched_mask * tf.expand_dims(weights, -1)) * 2.0
        return tf.reduce_sum(sq_err) / (n_valid + 1e-8)

    dist = tf.sqrt(tf.reduce_sum(tf.square(diff), axis=-1) + 1e-8)
    if LOCATION_LOSS == "huber":
        delta = tf.constant(LOCATION_HUBER_DELTA, dtype=tf.float32)
        per_slot = tf.where(
            dist <= delta,
            0.5 * tf.square(dist),
            delta * (dist - 0.5 * delta),
        )
    else:
        eps = tf.constant(LOCATION_CHARBONNIER_EPS, dtype=tf.float32)
        per_slot = eps * (tf.sqrt(tf.square(dist / eps) + 1.0) - 1.0)

    weighted = per_slot * matched_mask * tf.expand_dims(weights, -1)
    n_valid = tf.reduce_sum(matched_mask * tf.expand_dims(weights, -1))
    return tf.reduce_sum(weighted) / (n_valid + 1e-8)


def _compute_hungarian_loss(xy_true, mask_true, predictions, weights):
    predictions = list(predictions)
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

    xy_loss = _location_loss(pred_xy, matched_gt_xy, matched_mask, w)

    mask_loss = tf.constant(0.0)
    for i in range(4):
        m = matched_mask[:, i]
        mask_loss += _mask_bce(m, pred_presence[:, i], weights=w)
    mask_loss /= 4.0

    total_loss = xy_loss + MASK_LOSS_WEIGHT * mask_loss
    if COUNT_LOSS_WEIGHT > 0.0 and len(predictions) > 8:
        true_count = tf.cast(tf.reduce_sum(mask_true, axis=1), tf.int32)
        count_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=true_count, logits=predictions[8]
        )
        total_loss += COUNT_LOSS_WEIGHT * tf.reduce_mean(count_loss * w)

    return total_loss, xy_loss, mask_loss


compute_loss = _compute_hungarian_loss


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


def train_one_epoch_for_model(model_obj, optimizer_obj, dataset):
    losses, grad_norms = [], []
    for x_batch, y_batch in dataset:
        with tf.GradientTape() as tape:
            predictions = model_obj(x_batch, training=True)
            loss, _, _ = compute_loss(
                y_batch["xy"], y_batch["mask"], predictions, y_batch["weight"]
            )
        grads = tape.gradient(loss, model_obj.trainable_variables)
        grad_norm = tf.linalg.global_norm(grads)
        optimizer_obj.apply_gradients(zip(grads, model_obj.trainable_variables))
        losses.append(float(loss.numpy()))
        grad_norms.append(float(grad_norm.numpy()))
    return float(np.mean(losses)), float(np.mean(grad_norms))


def eval_epoch_for_model(model_obj, dataset):
    losses = []
    for x_batch, y_batch in dataset:
        predictions = model_obj(x_batch, training=False)
        loss, _, _ = compute_loss(
            y_batch["xy"], y_batch["mask"], predictions, y_batch["weight"]
        )
        losses.append(float(loss.numpy()))
    return float(np.mean(losses))


THRESHOLD_CANDIDATES = [round(x, 3) for x in np.arange(0.30, 0.86, 0.05)]


def _collect_predictions_for_model(model_obj, dataset):
    pred_xy_list, pred_pr_list, xy_true_list, mask_true_list = [], [], [], []
    for x_input, y_batch in dataset:
        preds = model_obj(x_input, training=False)
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


def _predict_mask_np(pred_pr, thresholds):
    thresholds = np.asarray(thresholds, dtype=np.float64)
    if thresholds.ndim == 0:
        thresholds = np.full(4, float(thresholds))
    return expit(pred_pr) > thresholds[None, :]


def _count_accuracy_np(pred_pr, mask_true, thresholds):
    mask_pred = _predict_mask_np(pred_pr, thresholds)
    pred_count = mask_pred.sum(axis=1)
    true_count = (mask_true > 0.5).sum(axis=1)
    return float(np.mean(pred_count == true_count))


def _tune_per_slot_thresholds_np(pred_pr, mask_true, candidates, iterations=2):
    thresholds = np.full(4, 0.5)
    for _ in range(iterations):
        for slot in range(4):
            best_t, best_acc = thresholds[slot], -1.0
            for t in candidates:
                test = thresholds.copy()
                test[slot] = t
                acc = _count_accuracy_np(pred_pr, mask_true, test)
                if acc > best_acc:
                    best_acc, best_t = acc, t
            thresholds[slot] = best_t
    return thresholds


def _metric_summary_np(pred_xy, pred_pr, xy_true, mask_true, thresholds):
    mask_pred = _predict_mask_np(pred_pr, thresholds)
    xy_errs = []
    count_ok = []
    for b in range(xy_true.shape[0]):
        cost = np.zeros((4, 4))
        for i in range(4):
            for j in range(4):
                if mask_true[b, j] > 0.5:
                    cost[i, j] = np.sum((pred_xy[b, i] - xy_true[b, j]) ** 2)
        _, col_ind = linear_sum_assignment(cost)
        for i, j in enumerate(col_ind):
            if mask_true[b, j] > 0.5:
                xy_errs.append(np.linalg.norm(pred_xy[b, i] - xy_true[b, j]))
        count_ok.append(int(mask_pred[b].sum()) == int(mask_true[b].sum()))
    loc_err = float(np.mean(xy_errs)) if xy_errs else float("nan")
    count_acc = float(np.mean(count_ok) * 100.0)
    return loc_err, count_acc


def _selection_score(location_error, count_accuracy):
    return location_error - TOPK_SELECTION_COUNT_WEIGHT * count_accuracy


def _output_sort_key(output_detail):
    name = output_detail["name"].lower()
    matches = re.findall(r"(\d+)", name)
    if matches:
        return int(matches[-1])
    return name


def _split_tflite_outputs(output_details):
    output_details = sorted(output_details, key=_output_sort_key)
    xy_out = [o for o in output_details if "xy" in o["name"].lower()]
    pr_out = [o for o in output_details if "presence" in o["name"].lower()]
    if len(xy_out) != 4 or len(pr_out) != 4:
        xy_out = [o for o in output_details if int(np.prod(o["shape"])) == 2]
        pr_out = [o for o in output_details if int(np.prod(o["shape"])) == 1]
    if len(xy_out) != 4 or len(pr_out) != 4:
        details = ", ".join(
            f"{o['name']} shape={tuple(int(x) for x in o['shape'])}"
            for o in output_details
        )
        raise RuntimeError(f"Could not identify xy/presence outputs: {details}")
    return xy_out, pr_out


def _representative_dataset():
    n_samples = min(500, len(train_features))
    for i in range(n_samples):
        yield [train_features[i : i + 1].astype(np.float32)]


def _convert_model_to_int8(model_obj):
    converter = tf.lite.TFLiteConverter.from_keras_model(model_obj)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = _representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    return converter.convert()


def _convert_current_model_to_int8():
    return _convert_model_to_int8(model)


def _make_qat_model(model_obj):
    quantizable = (
        tf.keras.layers.Conv2D,
        tf.keras.layers.DepthwiseConv2D,
        tf.keras.layers.SeparableConv2D,
        tf.keras.layers.Dense,
    )

    def annotate(layer):
        if isinstance(layer, quantizable):
            return tfmot.quantization.keras.quantize_annotate_layer(layer)
        return layer

    annotated_model = tf.keras.models.clone_model(model_obj, clone_function=annotate)
    annotated_model.set_weights(model_obj.get_weights())
    with tfmot.quantization.keras.quantize_scope(
        {
            "AntennaDropout": AntennaDropout,
            "RandomPhaseShift": RandomPhaseShift,
            "RadarBatchFlatten": RadarBatchFlatten,
            "ComplexFFTChannelSplit": ComplexFFTChannelSplit,
            "AmplitudeFFTChannelSplit": AmplitudeFFTChannelSplit,
            "RadarTokenReshape": RadarTokenReshape,
            "RadarPositionEmbedding": RadarPositionEmbedding,
            "MultiHeadRadarAttention": MultiHeadRadarAttention,
            "TopKAveragePooling2D": TopKAveragePooling2D,
            "SoftAttentionPooling2D": SoftAttentionPooling2D,
        }
    ):
        return tfmot.quantization.keras.quantize_apply(annotated_model)


def _predict_tflite_arrays(tflite_model, features):
    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_scale, input_zero_point = input_details[0]["quantization"]
    input_dtype = input_details[0]["dtype"]
    xy_out, pr_out = _split_tflite_outputs(output_details)

    pred_xy_list, pred_pr_list = [], []
    for idx in range(len(features)):
        sample = features[idx : idx + 1]
        if np.issubdtype(input_dtype, np.floating):
            model_input = sample.astype(input_dtype)
        else:
            info = np.iinfo(input_dtype)
            model_input = np.round(sample / input_scale + input_zero_point)
            model_input = np.clip(model_input, info.min, info.max).astype(input_dtype)
        interpreter.set_tensor(input_details[0]["index"], model_input)
        interpreter.invoke()

        sample_xy, sample_pr = [], []
        for o in xy_out:
            raw = interpreter.get_tensor(o["index"]).flatten()
            scale, zp = o["quantization"]
            sample_xy.append((raw.astype(np.float32) - zp) * scale)
        for o in pr_out:
            raw = interpreter.get_tensor(o["index"]).flatten()[0]
            scale, zp = o["quantization"]
            sample_pr.append((float(raw) - zp) * scale)
        pred_xy_list.append(np.stack(sample_xy, axis=0))
        pred_pr_list.append(np.array(sample_pr, dtype=np.float32))
    return np.stack(pred_xy_list, axis=0), np.stack(pred_pr_list, axis=0)


# ==========================================================================
# Main training loop
# ==========================================================================
print(
    f"\nTraining for {EPOCHS} epochs  (lr={LEARNING_RATE}, batch={BATCH_SIZE}, schedule={LR_SCHEDULE}, warmup={WARMUP_EPOCHS})"
)
print(
    f"FFT bins kept: [{fft_bin_lo}, {fft_bin_hi}]  freq kept: [{fft_freq_lo}, {fft_freq_hi}]"
)
print(
    f"Location loss: {LOCATION_LOSS} "
    f"(huber_delta={LOCATION_HUBER_DELTA}, charbonnier_eps={LOCATION_CHARBONNIER_EPS})"
)
if TOPK_PTQ_SELECTION:
    print(
        "Top-K PTQ selection enabled: "
        f"k={TOPK_PTQ_K}, eval_every={TOPK_PTQ_EVAL_EVERY}, "
        f"count_weight={TOPK_SELECTION_COUNT_WEIGHT}"
    )
print("-" * 80)

best_weights = None
best_val_loss = float("inf")
topk_candidates = []

for epoch in range(EPOCHS):
    train_losses, grad_norms = [], []
    for x_batch, y_batch in train_ds:
        loss, _, _, gn = train_step(
            x_batch, y_batch["xy"], y_batch["mask"], y_batch["weight"]
        )
        train_losses.append(loss.numpy())
        grad_norms.append(gn.numpy())

    val_loss = eval_epoch(val_ds)
    metric_msg = ""

    if TOPK_PTQ_SELECTION and (
        (epoch + 1) % TOPK_PTQ_EVAL_EVERY == 0 or (epoch + 1) == EPOCHS
    ):
        val_pred_for_selection = _collect_predictions_for_model(model, val_ds)
        ckpt_thresholds = _tune_per_slot_thresholds_np(
            val_pred_for_selection[1], val_pred_for_selection[3], THRESHOLD_CANDIDATES
        )
        ckpt_val_err, ckpt_val_acc = _metric_summary_np(
            *val_pred_for_selection, thresholds=ckpt_thresholds
        )
        ckpt_score = _selection_score(ckpt_val_err, ckpt_val_acc)
        candidate = {
            "epoch": epoch + 1,
            "val_loss": float(val_loss),
            "val_location_error": float(ckpt_val_err),
            "val_count_accuracy": float(ckpt_val_acc),
            "score": float(ckpt_score),
            "thresholds": ckpt_thresholds.copy(),
            "weights": model.get_weights(),
        }
        topk_candidates.append(candidate)
        topk_candidates = sorted(topk_candidates, key=lambda x: x["score"])[
            :TOPK_PTQ_K
        ]
        best_metric = topk_candidates[0]
        metric_msg = (
            f"  val_loc={ckpt_val_err:.4f}  val_cnt={ckpt_val_acc:.1f}%  "
            f"score={ckpt_score:.4f}  "
            f"top_epoch={best_metric['epoch']} top_score={best_metric['score']:.4f}"
        )

    print(
        f"Epoch {epoch + 1:3d}/{EPOCHS}  "
        f"train_loss={np.mean(train_losses):.4f}  "
        f"val_loss={val_loss:.4f}  "
        f"grad_norm={np.mean(grad_norms):.4f}  "
        f"lr={optimizer.learning_rate.numpy() if hasattr(optimizer.learning_rate, 'numpy') else float(optimizer.learning_rate):.6f}  "
        f"best_val={best_val_loss:.4f}"
        f"{metric_msg}"
    )

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_weights = model.get_weights()

if not TOPK_PTQ_SELECTION and best_weights is not None:
    model.set_weights(best_weights)
    print(f"\nRestored best model (val_loss={best_val_loss:.4f})")

if TOPK_PTQ_SELECTION:
    print("\nTop-K checkpoints before PTQ:")
    for rank, candidate in enumerate(topk_candidates, start=1):
        thr_str = "[" + ", ".join(f"{x:.2f}" for x in candidate["thresholds"]) + "]"
        print(
            f"  #{rank}: epoch={candidate['epoch']} "
            f"score={candidate['score']:.4f} "
            f"val_loc={candidate['val_location_error']:.4f} "
            f"val_cnt={candidate['val_count_accuracy']:.1f}% "
            f"val_loss={candidate['val_loss']:.4f} thresholds={thr_str}"
        )

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

        pred_count = int(mask_pred[b].sum())
        correct = true_count == pred_count
        count_accs.append(correct)
        count_accs_by_count[true_count].append(correct)
        count_confusion[true_count][pred_count] += 1

        true_xy = xy_true[b][mask_true[b] > 0.5]
        pred_xy_present = pred_xy[b][mask_pred[b]]

        if pred_count == true_count - 1 and true_count >= 1 and pred_count >= 1:
            min_dists_true = np.min(
                np.linalg.norm(
                    true_xy[:, None, :] - pred_xy_present[None, :, :], axis=-1
                ),
                axis=1,
            )
            missing_dist = float(np.max(min_dists_true))
            fusion_errs_by_count[true_count].append(missing_dist)
        elif pred_count == true_count + 1 and true_count >= 1 and pred_count <= 4:
            min_dists_pred = np.min(
                np.linalg.norm(
                    pred_xy_present[:, None, :] - true_xy[None, :, :], axis=-1
                ),
                axis=1,
            )
            fake_dist = float(np.max(min_dists_pred))
            splitting_errs_by_count[true_count].append(fake_dist)
        elif pred_count == true_count + 1 and true_count == 0 and pred_count <= 4:
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


def _predict_mask(pred_pr, thresholds):
    thresholds = np.asarray(thresholds, dtype=np.float64)
    if thresholds.ndim == 0:
        thresholds = np.full(4, float(thresholds))
    return expit(pred_pr) > thresholds[None, :]


def _count_accuracy(pred_pr, mask_true, thresholds):
    mask_pred = _predict_mask(pred_pr, thresholds)
    pred_count = mask_pred.sum(axis=1)
    true_count = (mask_true > 0.5).sum(axis=1)
    return float(np.mean(pred_count == true_count))


def tune_per_slot_thresholds(pred_pr, mask_true, candidates, iterations=2):
    thresholds = np.full(4, 0.5)
    for _ in range(iterations):
        for slot in range(4):
            best_t, best_acc = thresholds[slot], -1.0
            for t in candidates:
                test = thresholds.copy()
                test[slot] = t
                acc = _count_accuracy(pred_pr, mask_true, test)
                if acc > best_acc:
                    best_acc, best_t = acc, t
            thresholds[slot] = best_t
    return thresholds


def tune_thresholds(pred_pr, mask_true, threshold_candidates):
    print(f"\n=== Per-slot threshold tuning on validation ===")
    best_thresholds = tune_per_slot_thresholds(pred_pr, mask_true, threshold_candidates)
    acc = _count_accuracy(pred_pr, mask_true, best_thresholds) * 100
    thr_str = "[" + ", ".join(f"{x:.2f}" for x in best_thresholds) + "]"
    print(f"  -> best: thresholds={thr_str}  count_acc={acc:.1f}%")
    return best_thresholds


def evaluate_from_predictions(pred_xy, pred_pr, xy_true, mask_true, name, thresholds):
    all_xy_err = []
    all_count_correct = []
    xy_errs_by_count = {k: [] for k in range(5)}
    count_accs_by_count = {k: [] for k in range(5)}
    count_confusion = {k: {p: 0 for p in range(5)} for k in range(5)}
    fusion_errs_by_count = {k: [] for k in range(5)}
    splitting_errs_by_count = {k: [] for k in range(5)}

    mask_pred = _predict_mask(pred_pr, thresholds)
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
    header = (
        "         " + "true\\pred".rjust(12) + "".join(f"{p:>9d}" for p in range(5))
    )
    print(header)
    for k in range(5):
        total = sum(count_confusion[k].values())
        if total == 0:
            row = "         " + f"{k:>12d}" + "".join("      n/a" for _ in range(5))
        else:
            row = (
                "         "
                + f"{k:>12d}"
                + "".join(
                    f"{(count_confusion[k][p] / total * 100):>8.1f}%" for p in range(5)
                )
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
        fe = np.nanmean(fusion_errs_by_count[k]) if fn > 0 else float("nan")
        sn = len(splitting_errs_by_count[k])
        se = np.nanmean(splitting_errs_by_count[k]) if sn > 0 else float("nan")
        print(f"         {k:>10d}  {fn:>10d}  {fe:>12.4f}  {sn:>10d}  {se:>12.4f}")

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


selected_tflite_model = None
selected_ptq_candidate = None

if TOPK_PTQ_SELECTION:
    if not topk_candidates:
        raise RuntimeError("TOPK_PTQ_SELECTION enabled but no checkpoints were saved")

    print("\n=== Top-K PTQ selection on validation ===")
    ptq_candidates = []
    for rank, candidate in enumerate(topk_candidates, start=1):
        model.set_weights(candidate["weights"])
        print(
            f"  Quantizing candidate #{rank} from epoch {candidate['epoch']} "
            f"(pre-PTQ score={candidate['score']:.4f})"
        )
        candidate_tflite = _convert_current_model_to_int8()
        val_tflite_xy, val_tflite_pr = _predict_tflite_arrays(
            candidate_tflite, val_features
        )
        tflite_val_err, tflite_val_acc = _metric_summary_np(
            val_tflite_xy,
            val_tflite_pr,
            val_xy,
            val_mask,
            candidate["thresholds"],
        )
        tflite_score = _selection_score(tflite_val_err, tflite_val_acc)
        tflite_size_kb_candidate = len(candidate_tflite) / 1024
        print(
            f"    PTQ val_loc={tflite_val_err:.4f} "
            f"val_cnt={tflite_val_acc:.1f}% "
            f"score={tflite_score:.4f} "
            f"size={tflite_size_kb_candidate:.1f} KB"
        )
        ptq_candidate = dict(candidate)
        ptq_candidate.update(
            {
                "tflite_model": candidate_tflite,
                "tflite_val_location_error": float(tflite_val_err),
                "tflite_val_count_accuracy": float(tflite_val_acc),
                "tflite_score": float(tflite_score),
                "tflite_size_kb": float(tflite_size_kb_candidate),
            }
        )
        ptq_candidates.append(ptq_candidate)

    selected_ptq_candidate = min(ptq_candidates, key=lambda x: x["tflite_score"])
    selected_tflite_model = selected_ptq_candidate["tflite_model"]
    model.set_weights(selected_ptq_candidate["weights"])
    thr_str = "[" + ", ".join(f"{x:.2f}" for x in selected_ptq_candidate["thresholds"]) + "]"
    print(
        "Selected PTQ checkpoint: "
        f"epoch={selected_ptq_candidate['epoch']} "
        f"ptq_score={selected_ptq_candidate['tflite_score']:.4f} "
        f"ptq_val_loc={selected_ptq_candidate['tflite_val_location_error']:.4f} "
        f"ptq_val_cnt={selected_ptq_candidate['tflite_val_count_accuracy']:.1f}% "
        f"thresholds={thr_str}"
    )

quantization_type = "QAT" if QAT else "PTQ"

if QAT:
    print(
        "\n=== Quantization-aware fine-tuning ===\n"
        f"QAT epochs={QAT_EPOCHS} lr={QAT_LEARNING_RATE}"
    )
    model = _make_qat_model(model)
    optimizer = tf.keras.optimizers.Adam(QAT_LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    selected_tflite_model = None

    best_qat_weights = model.get_weights()
    best_qat_score = float("inf")
    best_qat_thresholds = None
    best_qat_epoch = 0

    for qat_epoch in range(QAT_EPOCHS):
        train_loss, grad_norm = train_one_epoch_for_model(model, optimizer, train_ds)
        val_loss = eval_epoch_for_model(model, val_ds)
        val_pred_for_selection = _collect_predictions_for_model(model, val_ds)
        qat_thresholds = _tune_per_slot_thresholds_np(
            val_pred_for_selection[1],
            val_pred_for_selection[3],
            THRESHOLD_CANDIDATES,
        )
        qat_val_err, qat_val_acc = _metric_summary_np(
            *val_pred_for_selection, thresholds=qat_thresholds
        )
        qat_score = _selection_score(qat_val_err, qat_val_acc)
        if qat_score < best_qat_score:
            best_qat_score = qat_score
            best_qat_weights = model.get_weights()
            best_qat_thresholds = qat_thresholds.copy()
            best_qat_epoch = qat_epoch + 1
        print(
            f"QAT epoch {qat_epoch + 1:3d}/{QAT_EPOCHS}  "
            f"train_loss={train_loss:.4f}  "
            f"val_loss={val_loss:.4f}  "
            f"grad_norm={grad_norm:.4f}  "
            f"val_loc={qat_val_err:.4f}  "
            f"val_cnt={qat_val_acc:.1f}%  "
            f"score={qat_score:.4f}  "
            f"best_qat_epoch={best_qat_epoch}"
        )

    if QAT_EPOCHS > 0:
        model.set_weights(best_qat_weights)
        thr_str = (
            "[" + ", ".join(f"{x:.2f}" for x in best_qat_thresholds) + "]"
            if best_qat_thresholds is not None
            else "n/a"
        )
        print(
            "Restored best QAT checkpoint: "
            f"epoch={best_qat_epoch} score={best_qat_score:.4f} "
            f"thresholds={thr_str}"
        )


print("\n=== Final Evaluation ===")
train_pred = _collect_predictions(train_ds)
val_pred = _collect_predictions(val_ds)
test_pred = _collect_predictions(test_ds)

if QAT and QAT_EPOCHS > 0 and best_qat_thresholds is not None:
    best_thresholds = best_qat_thresholds
    thr_str = "[" + ", ".join(f"{x:.2f}" for x in best_thresholds) + "]"
    print(f"\nUsing thresholds from selected QAT checkpoint: {thr_str}")
elif TOPK_PTQ_SELECTION and selected_ptq_candidate is not None:
    best_thresholds = selected_ptq_candidate["thresholds"]
    thr_str = "[" + ", ".join(f"{x:.2f}" for x in best_thresholds) + "]"
    print(f"\nUsing thresholds from selected PTQ checkpoint: {thr_str}")
else:
    best_thresholds = tune_thresholds(val_pred[1], val_pred[3], THRESHOLD_CANDIDATES)

print(f"\n--- Evaluation @ default threshold=0.500 ---")
evaluate_from_predictions(*train_pred, "Train", 0.5)
evaluate_from_predictions(*val_pred, "Val", 0.5)
evaluate_from_predictions(*test_pred, "Test", 0.5)

thr_str = "[" + ", ".join(f"{x:.2f}" for x in best_thresholds) + "]"
print(f"\n--- Evaluation @ tuned thresholds={thr_str} ---")
train_err, train_acc, train_bk = evaluate_from_predictions(
    *train_pred, "Train", best_thresholds
)
val_err, val_acc, val_bk = evaluate_from_predictions(*val_pred, "Val", best_thresholds)
test_err, test_acc, test_bk = evaluate_from_predictions(
    *test_pred, "Test", best_thresholds
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
    f"LR_SCHEDULE={LR_SCHEDULE} WARMUP_EPOCHS={WARMUP_EPOCHS} "
    f"MODEL=cplx_fft_dual_encoder_tokens D_ATTN={D_ATTN} N_HEADS={N_HEADS} "
    f"CONV_FILTERS={CONV_FILTERS} "
    f"COMPLEX_CONV_FILTERS={COMPLEX_CONV_FILTERS} "
    f"AMPLITUDE_CONV_FILTERS={AMPLITUDE_CONV_FILTERS} "
    f"CONV_KERNELS={CONV_KERNELS} "
    f"CONV_BLOCK_TYPE={CONV_BLOCK_TYPE} "
    f"MOBILENETV2_EXPANSION={MOBILENETV2_EXPANSION} "
    f"DENSE_DIM={DENSE_DIM} HEAD_DIM={HEAD_DIM} POS_EMBED={POS_EMBED} "
    f"FFT_BIN_RANGE={FFT_BIN_RANGE} FFT_FREQ_RANGE={FFT_FREQ_RANGE} "
    f"TRANS_BLOCKS={TRANS_BLOCKS} FFN_MULT={FFN_MULT} "
    f"ENCODER_POOLING={ENCODER_POOLING} ENCODER_TOPK={ENCODER_TOPK} "
    f"COMPLEX_ENCODER_POOLING={COMPLEX_ENCODER_POOLING} "
    f"AMPLITUDE_ENCODER_POOLING={AMPLITUDE_ENCODER_POOLING} "
    f"ENCODER_ATTN_TEMPERATURE={ENCODER_ATTN_TEMPERATURE} "
    f"COUNT_LOSS_WEIGHT={COUNT_LOSS_WEIGHT} "
    f"LOCATION_LOSS={LOCATION_LOSS} "
    f"LOCATION_HUBER_DELTA={LOCATION_HUBER_DELTA} "
    f"LOCATION_CHARBONNIER_EPS={LOCATION_CHARBONNIER_EPS} "
    f"TOPK_PTQ_SELECTION={TOPK_PTQ_SELECTION} TOPK_PTQ_K={TOPK_PTQ_K} "
    f"TOPK_PTQ_EVAL_EVERY={TOPK_PTQ_EVAL_EVERY} "
    f"TOPK_SELECTION_COUNT_WEIGHT={TOPK_SELECTION_COUNT_WEIGHT} "
    f"QAT={QAT} QAT_EPOCHS={QAT_EPOCHS} "
    f"QAT_LEARNING_RATE={QAT_LEARNING_RATE} "
    f"MACRO_WINDOW_SPLIT={MACRO_WINDOW_SPLIT} "
    f"AMPLITUDE_CHANNEL={AMPLITUDE_CHANNEL} "
    f"ANTENNA_DROPOUT={ANTENNA_DROPOUT} PHASE_SHIFT_MAX={PHASE_SHIFT_MAX} "
    f"MASK_LOSS_WEIGHT={MASK_LOSS_WEIGHT} "
    f"CLASS_BALANCE={CLASS_BALANCE} CLASS_LOSS_SCALE={CLASS_LOSS_SCALE}"
)

csv_path = os.path.join(os.path.dirname(__file__) or ".", "cplx_fft_results.csv")
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
model_dir = os.path.join(
    os.path.dirname(__file__) or ".", "exported_models", EXPERIMENT_NAME
)
os.makedirs(model_dir, exist_ok=True)
model.save(os.path.join(model_dir, "model.keras"))
print(f"Model exported to {model_dir}/model.keras")

# ==========================================================================
# INT8 Quantization & TFLite export
# ==========================================================================

if selected_tflite_model is not None:
    tflite_model = selected_tflite_model
    print("Using already-selected top-K PTQ TFLite model for final export")
else:
    tflite_model = _convert_current_model_to_int8()

interpreter = tf.lite.Interpreter(model_content=tflite_model)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_scale, input_zero_point = input_details[0]["quantization"]

xy_out, pr_out = _split_tflite_outputs(output_details)

keras_xy_err, tflite_xy_err = [], []
keras_count_ok, tflite_count_ok = [], []

for idx in range(len(test_features)):
    sample = test_features[idx : idx + 1]
    mask_true = test_mask[idx]
    xy_true = test_xy[idx]

    keras_out = model(sample, training=False)
    keras_xy = np.stack([keras_out[2 * i].numpy()[0] for i in range(4)], axis=0)
    keras_pr = np.stack([keras_out[2 * i + 1].numpy()[0, 0] for i in range(4)], axis=0)
    keras_mask_pred = _predict_mask(keras_pr[None, ...], best_thresholds)[0]

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
    tflite_mask_pred = _predict_mask(tflite_pr_arr[None, ...], best_thresholds)[0]

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

print(f"\n=== TFLite INT8 verification ({quantization_type}, full test set) ===")
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

model_params = {
    "mean": feat_mean,
    "std": feat_std,
    "baseline": baseline if baseline is not None else np.zeros_like(feat_mean),
    "has_baseline": np.array(baseline is not None),
    "fft_bin_lo": fft_bin_lo,
    "fft_bin_hi": fft_bin_hi,
    "fft_freq_lo": fft_freq_lo,
    "fft_freq_hi": fft_freq_hi,
    "sub_window_size": SUB_WINDOW_SIZE,
    "std_epsilon": STD_EPSILON,
    "thresholds": best_thresholds,
    "background_mode": BACKGROUND_MODE,
    "ema_alpha": EMA_ALPHA,
    "amplitude_channel": np.array(AMPLITUDE_CHANNEL),
}

np.savez(os.path.join(model_dir, "model_params"), **model_params)
print(f"Model params saved to {model_dir}/model_params.npz")

shutil.copy2(tflite_path, os.path.join(submission_dir, "model.tflite"))
print(f"TFLite model copied to {submission_dir}/model.tflite")

np.savez(os.path.join(submission_dir, "model_params"), **model_params)
print(f"Model params saved to {submission_dir}/model_params.npz")

# ==========================================================================
# Log quantization results to CSV
# ==========================================================================
q_csv_path = os.path.join(
    os.path.dirname(__file__) or ".", "cplx_fft_quant_results.csv"
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
    quantization_type,
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

print(f"{quantization_type} results appended to {q_csv_path}")

# ==========================================================================
# Export test metadata (for extract_test_data.py)
# ==========================================================================
test_metadata = {
    "seed": SEED,
    "sub_window_size": SUB_WINDOW_SIZE,
    "macro_window_split": MACRO_WINDOW_SPLIT,
    "test_sources": [macro_sources[i] for i in sorted(test_macro)],
    "n_test_macros": len(test_macro),
    "experiment_name": EXPERIMENT_NAME,
    "background_mode": BACKGROUND_MODE,
    "ema_alpha": EMA_ALPHA,
    "thresholds": best_thresholds.tolist(),
    "location_loss": LOCATION_LOSS,
    "location_huber_delta": LOCATION_HUBER_DELTA,
    "location_charbonnier_eps": LOCATION_CHARBONNIER_EPS,
    "topk_ptq_selection": TOPK_PTQ_SELECTION,
    "selected_epoch": int(selected_ptq_candidate["epoch"])
    if selected_ptq_candidate is not None
    else None,
    "selected_ptq_val_location_error": float(
        selected_ptq_candidate["tflite_val_location_error"]
    )
    if selected_ptq_candidate is not None
    else None,
    "selected_ptq_val_count_accuracy": float(
        selected_ptq_candidate["tflite_val_count_accuracy"]
    )
    if selected_ptq_candidate is not None
    else None,
}
metadata_path = os.path.join(model_dir, "test_metadata.json")
with open(metadata_path, "w") as f:
    json.dump(test_metadata, f, indent=2)
print(f"Test metadata saved to {metadata_path}")
