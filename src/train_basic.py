import os
import glob
import csv
import itertools
import numpy as np
import tensorflow as tf
from scipy.signal import wiener, savgol_filter
from scipy.optimize import linear_sum_assignment
from scipy.special import expit

# ==========================================================================
# Hyperparameters (from environment variables)
# ==========================================================================
SEED = int(os.environ.get("SEED", 42))
SUB_WINDOW_SIZE = int(os.environ.get("SUB_WINDOW_SIZE", 50))
TRAIN_RATIO = float(os.environ.get("TRAIN_RATIO", 0.7))
VAL_RATIO = float(os.environ.get("VAL_RATIO", 0.15))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 32))
EPOCHS = int(os.environ.get("EPOCHS", 10))
LEARNING_RATE = float(os.environ.get("LEARNING_RATE", 1e-3))
WEIGHT_DECAY = float(os.environ.get("WEIGHT_DECAY", 0.0))
DROPOUT = float(os.environ.get("DROPOUT", 0.0))
DATA_DIR = os.environ.get("DATA_DIR", "dataset/data")
STD_EPSILON = float(os.environ.get("STD_EPSILON", 0.0))
FILTER_TYPE = str(os.environ.get("FILTER_TYPE", "none"))
BASELINE_ZERO_PEOPLE = bool(int(os.environ.get("BASELINE_ZERO_PEOPLE", 0)))
HUNGARIAN_LOSS = bool(int(os.environ.get("HUNGARIAN_LOSS", 0)))
EXPERIMENT_NAME = os.environ.get("EXPERIMENT_NAME", "default")
SLIDING_WINDOW = bool(int(os.environ.get("SLIDING_WINDOW", 0)))
SEQ_LENGTH = int(os.environ.get("SEQ_LENGTH", 150))
SLIDING_STRIDE = int(os.environ.get("SLIDING_STRIDE", 5))

# ==========================================================================
# Reproducibility
# ==========================================================================
np.random.seed(SEED)
tf.random.set_seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)

# ==========================================================================
# Model definition
# ==========================================================================
N_RADARS = 6
N_ANTENNAS = 3
N_BINS = 120

inp = tf.keras.Input(
    shape=(SUB_WINDOW_SIZE, N_RADARS, N_ANTENNAS, N_BINS, 2), name="input"
)
x = tf.keras.layers.Reshape((SUB_WINDOW_SIZE, N_RADARS * N_ANTENNAS * N_BINS * 2))(inp)

for filters in [64, 128, 256]:
    x = tf.keras.layers.Conv1D(filters, kernel_size=3, padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPool1D(pool_size=2)(x)

x = tf.keras.layers.Flatten()(x)
x = tf.keras.layers.Dense(256)(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.ReLU()(x)
x = tf.keras.layers.Dropout(DROPOUT)(x)

outputs = []
for i in range(4):
    hi = tf.keras.layers.Dense(32, activation="relu")(x)
    xy_i = tf.keras.layers.Dense(2, name=f"p{i}_xy")(hi)
    presence_i = tf.keras.layers.Dense(1, name=f"p{i}_presence")(hi)
    outputs.extend([xy_i, presence_i])

model = tf.keras.Model(inputs=inp, outputs=outputs)
model.summary()

print(f"experiment: {EXPERIMENT_NAME}")

# ==========================================================================
# Data loading and preprocessing
# ==========================================================================
files = sorted(glob.glob(os.path.join(DATA_DIR, "window_*.npz")))
print(f"Found {len(files)} windows")

all_features = []
all_target_xy = []
all_target_mask = []

for fpath in files:
    d = np.load(fpath)
    iq = d["radar_cir_iq"]
    people_xy = d["people_xy"]
    people_mask = d["people_mask"]

    T = iq.shape[0]
    chunk_size = SEQ_LENGTH if SLIDING_WINDOW else SUB_WINDOW_SIZE
    n_sub = T // chunk_size
    usable = n_sub * chunk_size

    iq = iq[:usable]
    people_xy = people_xy[:usable]
    people_mask = people_mask[:usable]

    iq_sub = iq.reshape(n_sub, chunk_size, 6, 3, 120, 2)
    xy_sub = people_xy.reshape(n_sub, chunk_size, 4, 2)
    mask_sub = people_mask.reshape(n_sub, chunk_size, 4)

    amplitude = np.sqrt(iq_sub[..., 0] ** 2 + iq_sub[..., 1] ** 2)
    phase = np.arctan2(iq_sub[..., 1], iq_sub[..., 0])
    features = np.stack([amplitude, phase], axis=-1).astype(np.float32)

    if SLIDING_WINDOW:
        target_xy = xy_sub.astype(np.float32)
        target_mask = mask_sub.astype(np.float32)
    else:
        target_xy = xy_sub[:, -1, :, :].astype(np.float32)
        target_mask = mask_sub[:, -1, :].astype(np.float32)

    all_features.append(features)
    all_target_xy.append(target_xy)
    all_target_mask.append(target_mask)

features = np.concatenate(all_features, axis=0)
targets_xy = np.concatenate(all_target_xy, axis=0)
targets_mask = np.concatenate(all_target_mask, axis=0)

N = features.shape[0]
print(f"Total sub-windows: {N}  input shape: {features.shape}")

# ==========================================================================
# Shuffle and train/val/test split
# ==========================================================================
indices = np.arange(N)
np.random.shuffle(indices)

train_end = int(N * TRAIN_RATIO)
val_end = int(N * (TRAIN_RATIO + VAL_RATIO))

train_idx, val_idx, test_idx = (
    indices[:train_end],
    indices[train_end:val_end],
    indices[val_end:],
)

print(f"Split — train: {len(train_idx)}, val: {len(val_idx)}, test: {len(test_idx)}")

train_features, train_xy, train_mask = (
    features[train_idx],
    targets_xy[train_idx],
    targets_mask[train_idx],
)
val_features, val_xy, val_mask = (
    features[val_idx],
    targets_xy[val_idx],
    targets_mask[val_idx],
)
test_features, test_xy, test_mask = (
    features[test_idx],
    targets_xy[test_idx],
    targets_mask[test_idx],
)

# ==========================================================================
# Baseline subtraction (median of 0-people train sub-windows)
# ==========================================================================
if BASELINE_ZERO_PEOPLE:
    if SLIDING_WINDOW:
        zero_mask = train_mask[:, -1, :].sum(axis=1) == 0
    else:
        zero_mask = train_mask.sum(axis=1) == 0
    zero_train_idx = np.where(zero_mask)[0]
    if len(zero_train_idx) == 0:
        print("WARNING: no 0-people sub-windows in train set, skipping baseline")
    else:
        zero_feats = train_features[zero_train_idx].reshape(-1, 6, 3, 120, 2)
        bl_amp_med = np.median(zero_feats[:, :, :, :, 0], axis=0)
        bl_ph_med = np.median(zero_feats[:, :, :, :, 1], axis=0)
        train_features[:, :, :, :, :, 0] -= bl_amp_med
        train_features[:, :, :, :, :, 1] -= bl_ph_med
        val_features[:, :, :, :, :, 0] -= bl_amp_med
        val_features[:, :, :, :, :, 1] -= bl_ph_med
        test_features[:, :, :, :, :, 0] -= bl_amp_med
        test_features[:, :, :, :, :, 1] -= bl_ph_med
        print(
            f"Baseline subtracted from {len(zero_train_idx)} 0-people train sub-windows"
        )

# ==========================================================================
# Standardization (mean/std per radar, antenna, bin — computed on train set)
# ==========================================================================
flat = train_features.reshape(-1, 6, 3, 120, 2)
feat_mean = flat.mean(axis=0)
feat_std = flat.std(axis=0)
feat_std[feat_std == 0] = 1.0

train_features = (train_features - feat_mean) / (feat_std + STD_EPSILON)
val_features = (val_features - feat_mean) / (feat_std + STD_EPSILON)
test_features = (test_features - feat_mean) / (feat_std + STD_EPSILON)

# ==========================================================================
# Filtering
# ==========================================================================
if FILTER_TYPE == "wiener":
    for feats in [train_features, val_features, test_features]:
        for i in range(feats.shape[0]):
            feats[i] = wiener(feats[i], mysize=(15, 1, 1, 1, 1), noise=144)
elif FILTER_TYPE == "savgol":
    for feats in [train_features, val_features, test_features]:
        for i in range(feats.shape[0]):
            feats[i] = savgol_filter(feats[i], 10, 3, axis=0)

# ==========================================================================
# tf.data Datasets
# ==========================================================================
if SLIDING_WINDOW:
    _sample_spec = (
        tf.TensorSpec(
            shape=(SEQ_LENGTH, N_RADARS, N_ANTENNAS, N_BINS, 2), dtype=tf.float32
        ),
        {
            "xy": tf.TensorSpec(shape=(SEQ_LENGTH, 4, 2), dtype=tf.float32),
            "mask": tf.TensorSpec(shape=(SEQ_LENGTH, 4), dtype=tf.float32),
        },
    )
else:
    _sample_spec = (
        tf.TensorSpec(
            shape=(SUB_WINDOW_SIZE, N_RADARS, N_ANTENNAS, N_BINS, 2), dtype=tf.float32
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
# Fixed-window training (existing)
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


def eval_epoch_fixed(dataset):
    losses = []
    for x_batch, y_batch in dataset:
        predictions = model(x_batch, training=False)
        loss, _, _ = compute_loss(y_batch["xy"], y_batch["mask"], predictions)
        losses.append(loss.numpy())
    return np.mean(losses)


# ==========================================================================
# Sliding-window training (gradient accumulation)
# ==========================================================================
def _sliding_positions():
    return range(0, SEQ_LENGTH - SUB_WINDOW_SIZE + 1, SLIDING_STRIDE)


def train_epoch_sliding(dataset):
    losses = []
    for long_seq, y_batch in dataset:
        xy_all = y_batch["xy"]
        mask_all = y_batch["mask"]
        n_pos = len(list(_sliding_positions()))
        accum = [tf.zeros_like(v) for v in model.trainable_variables]
        batch_loss = 0.0

        for pos in _sliding_positions():
            x = long_seq[:, pos : pos + SUB_WINDOW_SIZE]
            xy_t = xy_all[:, pos + SUB_WINDOW_SIZE - 1]
            mask_t = mask_all[:, pos + SUB_WINDOW_SIZE - 1]

            with tf.GradientTape() as tape:
                preds = model(x, training=True)
                loss, _, _ = compute_loss(xy_t, mask_t, preds)
            grads = tape.gradient(loss, model.trainable_variables)
            accum = [a + g for a, g in zip(accum, grads)]
            batch_loss += loss.numpy()

        n = float(n_pos)
        accum = [a / n for a in accum]
        optimizer.apply_gradients(zip(accum, model.trainable_variables))
        losses.append(batch_loss / n)
    return np.mean(losses)


def eval_epoch_sliding(dataset):
    losses = []
    for long_seq, y_batch in dataset:
        xy_all = y_batch["xy"]
        mask_all = y_batch["mask"]
        n_pos = 0

        for pos in _sliding_positions():
            x = long_seq[:, pos : pos + SUB_WINDOW_SIZE]
            xy_t = xy_all[:, pos + SUB_WINDOW_SIZE - 1]
            mask_t = mask_all[:, pos + SUB_WINDOW_SIZE - 1]
            preds = model(x, training=False)
            loss, _, _ = compute_loss(xy_t, mask_t, preds)
            losses.append(loss.numpy())
            n_pos += 1
    return np.mean(losses)


# ==========================================================================
# Main training loop
# ==========================================================================
print(f"\nTraining for {EPOCHS} epochs  (lr={LEARNING_RATE}, batch={BATCH_SIZE})")
if SLIDING_WINDOW:
    print(
        f"Sliding window: seq={SEQ_LENGTH} sub={SUB_WINDOW_SIZE} stride={SLIDING_STRIDE}"
    )
    print(f"  positions per sequence: {len(list(_sliding_positions()))}")
print("-" * 80)

best_weights = None
best_val_loss = float("inf")

for epoch in range(EPOCHS):
    if SLIDING_WINDOW:
        train_loss = train_epoch_sliding(train_ds)
        val_loss = eval_epoch_sliding(val_ds)
        print(
            f"Epoch {epoch + 1:3d}/{EPOCHS}  "
            f"train_loss={train_loss:.4f}  "
            f"val_loss={val_loss:.4f}  "
            f"best_val={best_val_loss:.4f}"
        )
    else:
        train_losses, grad_norms = [], []
        for x_batch, y_batch in train_ds:
            loss, _, _, gn = train_step(x_batch, y_batch["xy"], y_batch["mask"])
            train_losses.append(loss.numpy())
            grad_norms.append(gn.numpy())

        val_loss = eval_epoch_fixed(val_ds)

        print(
            f"Epoch {epoch + 1:3d}/{EPOCHS}  "
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


def evaluate(dataset, name):
    all_xy_err = []
    all_count_correct = []

    for x_input, y_batch in dataset:
        xy_true = y_batch["xy"].numpy()
        mask_true = y_batch["mask"].numpy()

        if SLIDING_WINDOW:
            positions = list(_sliding_positions())
            for pos in positions:
                x = x_input[:, pos : pos + SUB_WINDOW_SIZE]
                xy_t = xy_true[:, pos + SUB_WINDOW_SIZE - 1]
                mask_t = mask_true[:, pos + SUB_WINDOW_SIZE - 1]
                preds = model(x, training=False)
                pred_xy = np.stack([preds[2 * i].numpy() for i in range(4)], axis=1)
                pred_pr = np.stack(
                    [preds[2 * i + 1].numpy()[:, 0] for i in range(4)], axis=1
                )
                mask_pred = expit(pred_pr) > 0.5
                _accumulate_metrics(
                    pred_xy,
                    mask_pred,
                    xy_t,
                    mask_t,
                    all_xy_err,
                    all_count_correct,
                )
        else:
            preds = model(x_input, training=False)
            pred_xy = np.stack([preds[2 * i].numpy() for i in range(4)], axis=1)
            pred_pr = np.stack(
                [preds[2 * i + 1].numpy()[:, 0] for i in range(4)], axis=1
            )
            mask_pred = expit(pred_pr) > 0.5
            _accumulate_metrics(
                pred_xy,
                mask_pred,
                xy_true,
                mask_true,
                all_xy_err,
                all_count_correct,
            )

    mean_err = np.mean(all_xy_err) if all_xy_err else float("nan")
    accuracy = np.mean(all_count_correct) * 100
    print(
        f"  {name:6s}  location_error={mean_err:.4f} m  count_accuracy={accuracy:.1f}%"
    )
    return mean_err, accuracy


def _accumulate_metrics(pred_xy, mask_pred, xy_true, mask_true, xy_errs, count_accs):
    for b in range(xy_true.shape[0]):
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
                    xy_errs.append(np.linalg.norm(pred_xy[b, i] - xy_true[b, j]))
        else:
            for j in range(4):
                if mask_true[b, j] > 0.5:
                    xy_errs.append(np.linalg.norm(pred_xy[b, j] - xy_true[b, j]))

        true_count = int(mask_true[b].sum())
        pred_count = int(mask_pred[b].sum())
        count_accs.append(true_count == pred_count)


print("\n=== Final Evaluation ===")
train_err, train_acc = evaluate(train_ds, "Train")
val_err, val_acc = evaluate(val_ds, "Val")
test_err, test_acc = evaluate(test_ds, "Test")

# ==========================================================================
# Log results to CSV
# ==========================================================================
hp_str = (
    f"SEED={SEED} SUB_WINDOW_SIZE={SUB_WINDOW_SIZE} "
    f"TRAIN_RATIO={TRAIN_RATIO} VAL_RATIO={VAL_RATIO} "
    f"BATCH_SIZE={BATCH_SIZE} EPOCHS={EPOCHS} "
    f"LEARNING_RATE={LEARNING_RATE} STD_EPSILON={STD_EPSILON} "
    f"FILTER_TYPE={FILTER_TYPE} BASELINE_ZERO_PEOPLE={BASELINE_ZERO_PEOPLE} "
    f"WEIGHT_DECAY={WEIGHT_DECAY} DROPOUT={DROPOUT} "
    f"HUNGARIAN_LOSS={HUNGARIAN_LOSS} "
    f"SLIDING_WINDOW={SLIDING_WINDOW} SEQ_LENGTH={SEQ_LENGTH} SLIDING_STRIDE={SLIDING_STRIDE}"
)

csv_path = os.path.join(os.path.dirname(__file__) or ".", "results.csv")
header = [
    "experiment_name",
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
