# Training scripts

The training scripts share a common configuration style: hyperparameters and architectural settings are supplied through environment variables. The final training pipeline is in `train.py`.

```bash
EXPERIMENT_NAME=final_model \
  DATA_DIR=<path-to-dataset> \
  SEED=42 \
  SUB_WINDOW_SIZE=50 \
  FFT_BIN_RANGE=5,48 \
  FFT_FREQ_RANGE=0,25 \
  TRAIN_RATIO=0.6 \
  VAL_RATIO=0.2 \
  BATCH_SIZE=16 \
  EPOCHS=200 \
  LEARNING_RATE=0.001 \
  STD_EPSILON=10.0 \
  EMA_ALPHA=0.96 \
  WEIGHT_DECAY=0.0005 \
  DROPOUT=0.3 \
  WARMUP_EPOCHS=5 \
  ENCODER_TOPK=6 \
  ENCODER_ATTN_TEMPERATURE=4.0 \
  COUNT_LOSS_WEIGHT=0.0 \
  LOCATION_CHARBONNIER_EPS=0.1 \
  TOPK_PTQ_K=5 \
  TOPK_PTQ_EVAL_EVERY=1 \
  TOPK_SELECTION_COUNT_WEIGHT=0.002 \
  MACRO_WINDOW_SPLIT=10 \
  ANTENNA_DROPOUT=0.1 \
  MASK_LOSS_WEIGHT=1.5 \
  python train.py
```

The final model uses a shared radar-level encoder. Complex FFT features and amplitude features are processed by separate convolutional branches; their per-radar representations are then fused by a lightweight attention layer before four location-and-presence prediction heads.

## Experiment families

### Radar encoder

- `radar_encoder/train_attn.py`: time-domain radar-level baseline.
- `radar_encoder/train_attn_fft.py`: FFT-amplitude radar-level model.
- `radar_encoder/train_cplx_fft.py`: complex FFT model, optionally with amplitude.
- `radar_encoder/train_cplx_fft_dual_encoder_fuse.py`: separate complex and amplitude encoders fused before attention.
- `radar_encoder/train_cplx_fft_dual_encoder_tokens.py`: separate complex and amplitude tokens passed to attention; the main precursor of the final architecture.
- `radar_encoder/train_cplx_fft_encoder_count_pretrain.py`, `train_cplx_fft_backbone_count_pretrain.py`, and `train_cplx_fft_count_aux_loss.py`: counting-oriented variants.

### Antenna encoder

- `antenna_encoder/train_ant_attn.py`: time-domain antenna-level baseline.
- `antenna_encoder/train_ant_mean_va.py`: amplitude mean/variance representation.
- `antenna_encoder/train_ant_fft.py`: FFT-amplitude antenna-level model.
- `antenna_encoder/train_ant_cplx_fft.py`: complex FFT antenna-level model.

The accompanying report describes the experiments, final configuration, and results in more detail.
