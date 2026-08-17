# TinyML UWB Multi-Person Localization

Joint project by Riccardo Masetti and Samuele Tondelli.

## Overview

This project explores on-device, privacy-preserving indoor localization with Ultra-Wideband (UWB) radar. The system estimates the two-dimensional positions of up to four people from raw complex Channel Impulse Response (CIR) measurements, without using cameras or wearable devices.

The final pipeline is designed for constrained edge deployment and exports an INT8 TensorFlow Lite model. A detailed account of the design process and experiments is available in [the project report](report/report.pdf).

## Approach

- Remove static reflections with exponential-moving-average background subtraction.
- Transform short temporal windows of complex I/Q radar signals into range-frequency maps with a temporal FFT.
- Process complex FFT components and amplitude in separate compact convolutional encoders.
- Fuse information from six radars with lightweight multi-head self-attention.
- Predict four unordered person slots, using Hungarian matching during training and a presence score for each slot.
- Select and export an INT8 TensorFlow Lite checkpoint for memory-efficient inference.

## Results

On a held-out test split, the final quantized model achieved:

| Metric | Result |
| --- | ---: |
| Average localization error | 0.4862 m |
| Count accuracy | 86.7% |
| F1 score | 0.8242 |
| TensorFlow Lite model size | 551.2 KB |
| Estimated activation arena | 266.6 KB |

## Repository layout

```text
src/            Training pipeline and architecture experiments
submission/     TensorFlow Lite model and inference entry point
report/         Technical project report
```

### Source code guide

The `src/` directory contains both the final training pipeline and the experiments that informed it:

- `src/train.py` is the final end-to-end pipeline. It loads and splits the data, applies preprocessing, trains the dual-encoder model, selects a post-training-quantized checkpoint, and exports the TensorFlow Lite artifact.
- `src/train_basic.py` is the initial time-domain Conv1D baseline, retained as a reference point for the later architectures.
- `src/radar_encoder/` contains the radar-level experiments. These progressively move from time-domain attention to FFT-based representations, complex I/Q features, and separate complex/amplitude encoders. `train_cplx_fft_dual_encoder_tokens.py` is the closest experimental predecessor of the final model.
- `src/antenna_encoder/` contains variants that first encode individual antennas before combining antenna and radar representations. These experiments explore whether preserving the antenna hierarchy improves localization.
- `src/README.md` provides the complete list of experiment scripts and their roles.

The inference path is kept separately in `submission/code.py`. It applies the same preprocessing used during training, reads normalization and threshold metadata from the exported model, runs the TensorFlow Lite interpreter, and writes predicted locations as JSON lines.

## Running inference

The inference entry point expects a NumPy array containing raw complex UWB CIR measurements with shape `(T, 6, 3, 120, 2)` and writes one JSON record per frame.

```bash
python submission/code.py \
  --input-path path/to/input.npy \
  --output-path path/to/output.jsonl
```

Install the required packages with:

```bash
pip install -r requirements.txt
```

## Data availability

The dataset used for this project is not included in this repository and cannot be redistributed. The repository is intended to document the model design and implementation.
