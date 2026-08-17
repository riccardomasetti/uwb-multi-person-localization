import os
import glob
import csv
import itertools
import numpy as np
import tensorflow as tf
from scipy.optimize import linear_sum_assignment
from scipy.special import expit

# ==========================================================================
# Hyperparameters (from environment variables)
# ==========================================================================
SEED = int(os.environ.get("SEED", 42))
SUB_WINDOW_SIZE = int(os.environ.get("SUB_WINDOW_SIZE", 50))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 32))
EPOCHS = int(os.environ.get("EPOCHS", 10))
LEARNING_RATE = float(os.environ.get("LEARNING_RATE", 1e-3))
WEIGHT_DECAY = float(os.environ.get("WEIGHT_DECAY", 0.0))
DROPOUT = float(os.environ.get("DROPOUT", 0.0))
DATA_DIR = os.environ.get("DATA_DIR", "dataset/data")
STD_EPSILON = float(os.environ.get("STD_EPSILON", 0.0))
BASELINE_ZERO_PEOPLE = bool(int(os.environ.get("BASELINE_ZERO_PEOPLE", 0)))
HUNGARIAN_LOSS = bool(int(os.environ.get("HUNGARIAN_LOSS", 0)))
EXPERIMENT_NAME = os.environ.get("EXPERIMENT_NAME", "mean_var_default")
CONV_FILTERS = os.environ.get("CONV_FILTERS", "32,48")
CONV_KERNELS = os.environ.get("CONV_KERNELS", "5,3")
DENSE_DIM = int(os.environ.get("DENSE_DIM", 256))
HEAD_DIM = int(os.environ.get("HEAD_DIM", 32))
LR_SCHEDULE = os.environ.get("LR_SCHEDULE", "constant")

conv_filters = [int(x) for x in CONV_FILTERS.split(",")]
conv_kernels = [int(x) for x in CONV_KERNELS.split(",")]
assert len(conv_filters) == len(conv_kernels), (
    "CONV_FILTERS and CONV_KERNELS must have same length"
)

# ==========================================================================
# Reproducibility
# ==========================================================================
np.random.seed(SEED)
tf.random.set_seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)

# ==========================================================================
# Model definition: Dual-stream 1D CNN (mean + variance) → FFN → 4 heads
# ==========================================================================
N_RADARS = 6
N_ANTENNAS = 3
N_BINS = 120
N_CHANNELS = 2  # mean and variance
N_ANTENNAS_TOTAL = N_RADARS * N_ANTENNAS  # 18


def _build_stream(filters, kernels, stream_name):
    """Build a 1D CNN stream for either mean or variance."""
    inputs = tf.keras.Input(shape=(N_BINS, 1), name=f"{stream_name}_input")
    x = inputs
    for i, (f, k) in enumerate(zip(filters, kernels)):
        x = tf.keras.layers.Conv1D(f, k, padding="same", name=f"{stream_name}_conv{i}")(x)
        x = tf.keras.layers.BatchNormalization(name=f"{stream_name}_bn{i}")(x)
        x = tf.keras.layers.ReLU(name=f"{stream_name}_relu{i}")(x)
        x = tf.keras.layers.MaxPool1D(pool_size=2, name=f"{stream_name}_pool{i}")(x)
    x = tf.keras.layers.GlobalAveragePooling1D(name=f"{stream_name}_gap")(x)
    return tf.keras.Model(inputs, x, name=f"{stream_name}_stream")


# Build shared streams (same weights for all antennas)
mean_stream = _build_stream(conv_filters, conv_kernels, "mean")
var_stream = _build_stream(conv_filters, conv_kernels, "var")

# Main model input
inp = tf.keras.Input(
    shape=(N_RADARS, N_ANTENNAS, N_BINS, N_CHANNELS), name="input"
)

# Split mean and variance
mean_input = tf.keras.layers.Lambda(
    lambda t: tf.reshape(t[..., 0], [-1, N_BINS, 1]),
    name="reshape_mean"
)(inp)  # (B*18, 120, 1)

var_input = tf.keras.layers.Lambda(
    lambda t: tf.reshape(t[..., 1], [-1, N_BINS, 1]),
    name="reshape_var"
)(inp)  # (B*18, 120, 1)

# Process through dual streams (shared weights across all antennas)
mean_features = mean_stream(mean_input)  # (B*18, last_conv_dim)
var_features = var_stream(var_input)     # (B*18, last_conv_dim)

# Concatenate mean and variance features per antenna
features = tf.keras.layers.Concatenate(name="fuse_mean_var")(
    [mean_features, var_features]
)  # (B*18, 2*last_conv_dim)

# Reshape back to per-sample: (B, 18, fused_dim)
fused_dim = 2 * conv_filters[-1]
features = tf.keras.layers.Lambda(
    lambda t: tf.reshape(t, [-1, N_ANTENNAS_TOTAL, fused_dim]),
    name="reshape_antennas"
)(features)  # (B, 18, fused_dim)

# Flatten all antennas into a single feature vector
features = tf.keras.layers.Flatten(name="flatten_antennas")(features)
# (B, 18 * fused_dim)

# FFN head
x = tf.keras.layers.Dense(DENSE_DIM, name="dense1")(features)
x = tf.keras.layers.BatchNormalization(name="bn1")(x)
x = tf.keras.layers.ReLU(name="relu1")(x)
x = tf.keras.layers.Dropout(DROPOUT, name="dropout")(x)

# 4 independent output heads
outputs = []
for i in range(4):
    hi = tf.keras.layers.Dense(HEAD_DIM, activation="relu", name=f"head{i}_dense")(x)
    xy_i = tf.keras.layers.Dense(2, name=f"p{i}_xy")(hi)
    presence_i = tf.keras.layers.Dense(1, name=f"p{i}_presence")(hi)
    outputs.extend([xy_i, presence_i])

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
# Data loading and preprocessing (window-level split)
# ==========================================================================
# Split ratio: 62.5% train / 18.75% val / 18.75% test (15 / 4.5 / 4.5 windows)
#
# TRAIN (15 windows):
#   0 people: 22          | 1 person: 16, 17, 18, 19
#   2 people: 0, 1, 2     | 3 people: 12, 13
#   4 people: 5, 6, 7, 8, 9
#
# VAL (4 windows + first half of 23):
#   0 people: 23 (first 3750 frames)
#   1 person: 20          | 2 people: 3
#   3 people: 14          | 4 people: 10
#
# TEST (4 windows + second half of 23):
#   0 people: 23 (last 3750 frames)
#   1 person: 21          | 2 people: 4
#   3 people: 15          | 4 people: 11
# ==========================================================================
TRAIN_WINDOWS = [22, 16, 17, 18, 19, 0, 1, 2, 12, 13, 5, 6, 7, 8, 9]
VAL_WINDOWS   = [20, 3, 14, 10]
TEST_WINDOWS  = [21, 4, 15, 11]


def _process_window(iq, people_xy, people_mask):
    T = iq.shape[0]
    chunk_size = SUB_WINDOW_SIZE
    n_sub = T // chunk_size
    usable = n_sub * chunk_size

    iq = iq[:usable]
    people_xy = people_xy[:usable]
    people_mask = people_mask[:usable]

    iq_sub = iq.reshape(n_sub, chunk_size, 6, 3, 120, 2)
    xy_sub = people_xy.reshape(n_sub, chunk_size, 4, 2)
    mask_sub = people_mask.reshape(n_sub, chunk_size, 4)

    amplitude = np.sqrt(iq_sub[..., 0] ** 2 + iq_sub[..., 1] ** 2)
    # shape: (n_sub, chunk_size, 6, 3, 120)

    # Compute mean and variance along time axis (axis=1)
    amp_mean = amplitude.mean(axis=1)
    amp_var = amplitude.var(axis=1)
    # shape: (n_sub, 6, 3, 120) each

    # Stack mean and variance as the last dimension
    features = np.stack([amp_mean, amp_var], axis=-1).astype(np.float32)
    # shape: (n_sub, 6, 3, 120, 2)

    target_xy = xy_sub[:, -1, :, :].astype(np.float32)
    target_mask = mask_sub[:, -1, :].astype(np.float32)

    return features, target_xy, target_mask


train_features_list, train_xy_list, train_mask_list = [], [], []
val_features_list, val_xy_list, val_mask_list = [], [], []
test_features_list, test_xy_list, test_mask_list = [], [], []

for w in TRAIN_WINDOWS:
    fpath = os.path.join(DATA_DIR, f"window_{w:06d}.npz")
    d = np.load(fpath)
    f, xy, m = _process_window(d["radar_cir_iq"], d["people_xy"], d["people_mask"])
    train_features_list.append(f)
    train_xy_list.append(xy)
    train_mask_list.append(m)

for w in VAL_WINDOWS:
    fpath = os.path.join(DATA_DIR, f"window_{w:06d}.npz")
    d = np.load(fpath)
    f, xy, m = _process_window(d["radar_cir_iq"], d["people_xy"], d["people_mask"])
    val_features_list.append(f)
    val_xy_list.append(xy)
    val_mask_list.append(m)

for w in TEST_WINDOWS:
    fpath = os.path.join(DATA_DIR, f"window_{w:06d}.npz")
    d = np.load(fpath)
    f, xy, m = _process_window(d["radar_cir_iq"], d["people_xy"], d["people_mask"])
    test_features_list.append(f)
    test_xy_list.append(xy)
    test_mask_list.append(m)

# Window 23 special handling: first 3750 frames to val, last 3750 frames to test
fpath = os.path.join(DATA_DIR, "window_000023.npz")
d = np.load(fpath)
f_val, xy_val, m_val = _process_window(
    d["radar_cir_iq"][:3750],
    d["people_xy"][:3750],
    d["people_mask"][:3750],
)
val_features_list.append(f_val)
val_xy_list.append(xy_val)
val_mask_list.append(m_val)

f_test, xy_test, m_test = _process_window(
    d["radar_cir_iq"][3750:7500],
    d["people_xy"][3750:7500],
    d["people_mask"][3750:7500],
)
test_features_list.append(f_test)
test_xy_list.append(xy_test)
test_mask_list.append(m_test)

train_features = np.concatenate(train_features_list, axis=0)
train_xy = np.concatenate(train_xy_list, axis=0)
train_mask = np.concatenate(train_mask_list, axis=0)

val_features = np.concatenate(val_features_list, axis=0)
val_xy = np.concatenate(val_xy_list, axis=0)
val_mask = np.concatenate(val_mask_list, axis=0)

test_features = np.concatenate(test_features_list, axis=0)
test_xy = np.concatenate(test_xy_list, axis=0)
test_mask = np.concatenate(test_mask_list, axis=0)

# Shuffle within each split independently
def _shuffle_split(features, xy, mask, seed_offset):
    np.random.seed(SEED + seed_offset)
    idx = np.arange(features.shape[0])
    np.random.shuffle(idx)
    return features[idx], xy[idx], mask[idx]

train_features, train_xy, train_mask = _shuffle_split(train_features, train_xy, train_mask, 0)
val_features, val_xy, val_mask = _shuffle_split(val_features, val_xy, val_mask, 1)
test_features, test_xy, test_mask = _shuffle_split(test_features, test_xy, test_mask, 2)

print(f"Split — train: {train_features.shape[0]}, val: {val_features.shape[0]}, test: {test_features.shape[0]}")
print(f"Input shape: {train_features.shape[1:]}")

# ==========================================================================
# Baseline subtraction (median of 0-people train sub-windows)
# ==========================================================================
if BASELINE_ZERO_PEOPLE:
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
            f"Baseline subtracted from {len(zero_train_idx)} 0-people train sub-windows"
        )

# ==========================================================================
# Standardization (mean/std per radar, antenna, bin, channel — computed on train set)
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
        shape=(N_RADARS, N_ANTENNAS, N_BINS, N_CHANNELS), dtype=tf.float32
    ),
    {
        "xy": tf.TensorSpec(shape=(4, 2), dtype=tf.float32),
        "mask": tf.TensorSpec(shape=(4,), dtype=tf.float32),
    },
)


def _make_generator(features, xy, mask):
    def gen():
        for i in range(features.shape[0]):
            yield features[i], {"xy": xy[i], "mask": mask[i]}

    return gen


train_ds = (
    tf.data.Dataset.from_generator(
        _make_generator(train_features, train_xy, train_mask),
        output_signature=_sample_spec,
    )
    .shuffle(1024, seed=SEED)
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)
val_ds = (
    tf.data.Dataset.from_generator(
        _make_generator(val_features, val_xy, val_mask), output_signature=_sample_spec
    )
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)
test_ds = (
    tf.data.Dataset.from_generator(
        _make_generator(test_features, test_xy, test_mask),
        output_signature=_sample_spec,
    )
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)


# ==========================================================================
# Loss and training loop
# ==========================================================================
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


def _compute_fixed_loss(xy_true, mask_true, predictions):
    xy_loss = tf.constant(0.0)
    mask_loss = tf.constant(0.0)
    for i in range(4):
        xy_pred = predictions[2 * i]
        presence_logits = predictions[2 * i + 1]
        m = mask_true[:, i]
        m_exp = tf.expand_dims(m, -1)
        sq_err = tf.square(xy_pred - xy_true[:, i, :]) * m_exp
        n_valid = tf.reduce_sum(m) * 2.0 + 1e-8
        xy_loss += tf.reduce_sum(sq_err) / n_valid
        mask_loss += tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                labels=m, logits=presence_logits[:, 0]
            )
        )
    return xy_loss + mask_loss, xy_loss, mask_loss


def _compute_hungarian_loss(xy_true, mask_true, predictions):
    pred_xy = tf.stack([predictions[2 * i] for i in range(4)], axis=1)
    pred_presence = tf.stack([predictions[2 * i + 1][:, 0] for i in range(4)], axis=1)

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
    sq_err = tf.square(pred_xy - matched_gt_xy) * mask_exp
    n_valid = tf.reduce_sum(matched_mask) * 2.0 + 1e-8
    xy_loss = tf.reduce_sum(sq_err) / n_valid

    mask_loss = tf.constant(0.0)
    for i in range(4):
        m = matched_mask[:, i]
        mask_loss += tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                labels=m, logits=pred_presence[:, i]
            )
        )
    mask_loss /= 4.0

    return xy_loss + mask_loss, xy_loss, mask_loss


compute_loss = _compute_hungarian_loss if HUNGARIAN_LOSS else _compute_fixed_loss


# ==========================================================================
# Training
# ==========================================================================
@tf.function
def train_step(x, xy_true, mask_true):
    with tf.GradientTape() as tape:
        predictions = model(x, training=True)
        loss, xy_l, mask_l = compute_loss(xy_true, mask_true, predictions)
    grads = tape.gradient(loss, model.trainable_variables)
    grad_norm = tf.linalg.global_norm(grads)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss, xy_l, mask_l, grad_norm


def eval_epoch(dataset):
    losses = []
    for x_batch, y_batch in dataset:
        predictions = model(x_batch, training=False)
        loss, _, _ = compute_loss(y_batch["xy"], y_batch["mask"], predictions)
        losses.append(loss.numpy())
    return np.mean(losses)


# ==========================================================================
# Learning rate schedule
# ==========================================================================
WARMUP_EPOCHS = 5

def get_lr(epoch):
    if LR_SCHEDULE == "constant":
        return LEARNING_RATE
    elif LR_SCHEDULE == "warmup_cosine":
        if epoch < WARMUP_EPOCHS:
            return LEARNING_RATE * (epoch + 1) / WARMUP_EPOCHS
        else:
            progress = (epoch - WARMUP_EPOCHS) / (EPOCHS - WARMUP_EPOCHS)
            return LEARNING_RATE * 0.5 * (1 + np.cos(np.pi * progress))
    elif LR_SCHEDULE == "step_decay":
        if epoch < 100:
            return LEARNING_RATE
        else:
            return LEARNING_RATE * 0.1
    else:
        return LEARNING_RATE


# ==========================================================================
# Main training loop
# ==========================================================================
print(f"\nTraining for {EPOCHS} epochs  (lr={LEARNING_RATE}, batch={BATCH_SIZE}, schedule={LR_SCHEDULE})")
print(f"Input shape: (N_RADARS={N_RADARS}, N_ANTENNAS={N_ANTENNAS}, N_BINS={N_BINS}, N_CHANNELS={N_CHANNELS})")
print(f"Dual-stream 1D CNN: {conv_filters} filters, kernels {conv_kernels}")
print("-" * 80)

best_weights = None
best_val_loss = float("inf")

for epoch in range(EPOCHS):
    current_lr = get_lr(epoch)
    optimizer.learning_rate.assign(current_lr)
    
    train_losses, grad_norms = [], []
    for x_batch, y_batch in train_ds:
        loss, _, _, gn = train_step(x_batch, y_batch["xy"], y_batch["mask"])
        train_losses.append(loss.numpy())
        grad_norms.append(gn.numpy())

    val_loss = eval_epoch(val_ds)

    print(
        f"Epoch {epoch + 1:3d}/{EPOCHS}  "
        f"lr={current_lr:.6f}  "
        f"train_loss={np.mean(train_losses):.4f}  "
        f"val_loss={val_loss:.4f}  "
        f"grad_norm={np.mean(grad_norms):.4f}  "
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


def _accumulate_metrics(pred_xy, mask_pred, xy_true, mask_true, metrics):
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
                    metrics["loc"][true_count].append(
                        np.linalg.norm(pred_xy[b, i] - xy_true[b, j])
                    )
        else:
            for j in range(4):
                if mask_true[b, j] > 0.5:
                    metrics["loc"][true_count].append(
                        np.linalg.norm(pred_xy[b, j] - xy_true[b, j])
                    )

        pred_count = int(mask_pred[b].sum())
        metrics["count"][true_count].append(true_count == pred_count)


def evaluate(dataset, name):
    metrics = {"loc": {c: [] for c in range(5)}, "count": {c: [] for c in range(5)}}

    for x_input, y_batch in dataset:
        xy_true = y_batch["xy"].numpy()
        mask_true = y_batch["mask"].numpy()

        preds = model(x_input, training=False)
        pred_xy = np.stack([preds[2 * i].numpy() for i in range(4)], axis=1)
        pred_pr = np.stack([preds[2 * i + 1].numpy()[:, 0] for i in range(4)], axis=1)
        mask_pred = expit(pred_pr) > 0.5
        _accumulate_metrics(pred_xy, mask_pred, xy_true, mask_true, metrics)

    # Overall
    all_xy_err = [err for c in range(5) for err in metrics["loc"][c]]
    all_count_correct = [acc for c in range(5) for acc in metrics["count"][c]]
    mean_err = np.mean(all_xy_err) if all_xy_err else float("nan")
    accuracy = np.mean(all_count_correct) * 100
    print(
        f"  {name:6s}  location_error={mean_err:.4f} m  count_accuracy={accuracy:.1f}%"
    )

    # Per-count breakdown
    for c in range(5):
        c_errs = metrics["loc"][c]
        c_accs = metrics["count"][c]
        c_loc = np.mean(c_errs) if c_errs else float("nan")
        c_count = np.mean(c_accs) * 100 if c_accs else float("nan")
        n_samples = len(c_accs)
        print(
            f"    {c} people:  location_error={c_loc:.4f} m  "
            f"count_accuracy={c_count:.1f}%  (n={n_samples})"
        )

    return mean_err, accuracy


print("\n=== Final Evaluation ===")
train_err, train_acc = evaluate(train_ds, "Train")
val_err, val_acc = evaluate(val_ds, "Val")
test_err, test_acc = evaluate(test_ds, "Test")

# Summary table
print("\n=== Summary ===")
print(f"  {'Split':<6s}  {'location_error':>14s}  {'count_accuracy':>14s}")
print("  " + "-" * 36)
print(f"  {'Train':<6s}  {train_err:>10.4f} m  {train_acc:>12.1f}%")
print(f"  {'Val':<6s}  {val_err:>10.4f} m  {val_acc:>12.1f}%")
print(f"  {'Test':<6s}  {test_err:>10.4f} m  {test_acc:>12.1f}%")

# ==========================================================================
# Log results to CSV
# ==========================================================================
hp_str = (
    f"SEED={SEED} SUB_WINDOW_SIZE={SUB_WINDOW_SIZE} "
    f"SPLIT=window_level "
    f"BATCH_SIZE={BATCH_SIZE} EPOCHS={EPOCHS} "
    f"LEARNING_RATE={LEARNING_RATE} STD_EPSILON={STD_EPSILON} "
    f"BASELINE_ZERO_PEOPLE={BASELINE_ZERO_PEOPLE} "
    f"WEIGHT_DECAY={WEIGHT_DECAY} DROPOUT={DROPOUT} "
    f"HUNGARIAN_LOSS={HUNGARIAN_LOSS} "
    f"MODEL=mean_var DENSE_DIM={DENSE_DIM} HEAD_DIM={HEAD_DIM} "
    f"CONV_FILTERS={CONV_FILTERS} CONV_KERNELS={CONV_KERNELS} "
    f"LR_SCHEDULE={LR_SCHEDULE}"
)

csv_path = os.environ.get("RESULTS_CSV", os.path.join(os.path.dirname(__file__) or ".", "mean_var_results.csv"))
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
