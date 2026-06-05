# Deepfake Detection via Optical Flow Residuals and CNN-LSTM

**Status:** Exploratory one-week module project, paused at planning/early-implementation stage. Documented here as a research log rather than a finished system.

## Overview

This project investigates **deepfake video detection through the structure of temporal coherence** rather than per-frame spatial artefacts. A statistical test for residual independance/entropy. The premise is that synthetically generated and physically captured video differ in a way that is not artefact-based but *structural*. Generated video is produced by a deterministic procedure operating on a latent representation, and its frame-to-frame coherence, achieved by interpolation in latent space, by attention bridging frames, or by explicit temporal-consistency losses, leaving structured, non-independent patterns in the residuals between frames. Genuine capture carries the high-entropy noise of a real sensor instead. So even when the visible output is convincing, the inter-frame residuals are not independent in the way real sensor noise is, and the hypothesis is that this structural difference is measurable.

The approach combines:

- **Farneback dense optical flow** computed dynamically between consecutive frames
- A **CNN-LSTM hybrid model** (EfficientNet-B0 backbone feeding into an LSTM) to capture spatial features and temporal irregularities respectively
- **Bayesian Optimization** for hyperparameter tuning, given limited compute

The project pivoted away from an initial PCA / statistical feature engineering plan once dataset handling proved more timeconsuming than expected.

## Documentation

This README is the project overview. The full write-up lives in **[METHODOLOGY.md](METHODOLOGY.md)**, which covers the problem framing (§1), a review of the related detection literature and where this project sits within it (§2 — Related Work), and the reference list. Read the README for the shape of the project; read the methodology for the detail and the citations.

## Why Optical Flow

Most published deepfake detectors operate on individual frames and learn spatial inconsistencies — eye reflections, skin texture, frequency-domain artefacts. These methods report strong numbers on clean benchmarks but degrade sharply on in-the-wild content. The structural signal is complementary, and harder to erase, because it lives not in any single frame but in the relationship *between* frames: a generative procedure's deterministic, non-independent inter-frame structure should differ from the high-entropy noise of genuine capture even when every individual frame is convincing.

Optical flow estimates that inter-frame relationship directly, making the structure in the residuals explicit and visualisable.

## The Compression Problem

A central limitation of optical-flow-based detection — and one that became clear during this project — is **lossy video compression**.

Academic deepfake datasets typically provide high-bitrate source video. Real-world deepfakes circulate through YouTube, TikTok, WhatsApp, and other platforms that apply aggressive H.264/H.265 re-encoding. This compression specifically targets and smooths high-frequency temporal residuals — which is exactly where optical-flow-based detection finds its signal. The codec's own motion estimation effectively launders the artefacts the detector is trying to see. 

Practically, this means:

- Heatmaps that cleanly highlight manipulated regions on uncompressed ground-truth video become near-indistinguishable noise on the same video after social-media re-encoding
- Detection accuracy on clean benchmarks systematically overstates real-world performance
- Any deployable detector needs to be evaluated against the actual distribution channel, not the lab dataset

This is a known but under-emphasised gap in the literature and would be a primary focus of any continuation of this work.

## The Human-in-the-Loop Problem

A second, deeper limitation: detection methods of this kind assume the adversary is a *pipeline* — a generation model whose statistical fingerprints can be learned. They are far less robust against a *human collaborator* working in post-production.

A skilled filmmaker or colourist treating generated footage as raw material — regrading, matching lens characteristics, reintroducing realistic motion blur and grain, simulating a plausible compression and re-encoding history, fixing temporal inconsistencies by hand  The output no longer looks like generator output, because it has been processed through the same finishing chain as real footage.

Optical flow residuals, frequency-domain artefacts, and most other statistical detection signals are designed for the unprocessed-pipeline case and degrade substantially in the post-produced case. This suggests detection alone is unlikely to be a long-term solution; cryptographic provenance (e.g. C2PA) and capture-time attestation are increasingly considered the more sustainable path. (The fuller version of this argument — why high-stakes content is exactly where detection is weakest, and why provenance is the more credible long-term answer is developed in METHODOLOGY.md §1.4 Limits of Detection and §1.5 Stakes and Provenance)

This project was undertaken with that limitation in view. It is offered as an exploration of one detection signal, not as a claim that detection of skilled fakes is achievable in general.

## Pipeline

1. **Frame extraction** — videos decoded into frame sequences, stored in video-named subdirectories
2. **Sequence normalisation** — all sequences truncated to the shortest video length
3. **Split** — 80/20 train/test, plus a holdout set (1 sequence per class)
4. **Optical flow** — Farneback dense flow computed on-the-fly during training; vectors converted to grayscale magnitude maps and stacked as additional input channels
5. **Model** — EfficientNet-B0 spatial features → LSTM temporal modelling → binary classification
6. **Tuning** — Bayesian Optimization over learning rate, batch size, LSTM hidden units
7. **Evaluation** — Accuracy, Precision, Recall, AUC-ROC; baseline comparison against CNN-only classifier

## Dataset

[Deep Fake Detection (DFD) dataset](https://www.kaggle.com/datasets/sanikatiwarekar/deep-fake-detection-dfd-entire-original-dataset) on Kaggle.

**Dataset critique:** In retrospect this dataset is a poor choice for a project intended to generalise. It predates current generation methods (diffusion-based face synthesis, more recent autoregressive video models) and contains compression and resolution characteristics that don't reflect contemporary in-the-wild deepfakes. A redo would use a more recent and more heterogeneous source, ideally including platform-re-encoded examples — candidates include FaceForensics++, the Deepfake Detection Challenge (DFDC) set, Celeb-DF, FaceShifter, DeeperForensics, and TrustedMedia (TM).

## Repository Structure

```
DeepFake-Detection-Temporal-Coherence/
├── data/                       # Dataset references and small samples
├── notebooks/                  # Jupyter / Colab notebooks
├── src/                        # Source scripts
├── results/                    # Checkpoints and metrics
├── TESSA_EDWARDS_DFD_V1.ipynb  # Main analysis notebook
├── README.md
├── METHODOLOGY.md              # Full write-up: problem framing, related work, references
├── requirements.txt
├── model_card.md               # Architecture and performance details
├── data_card.md                # Dataset details
├── LICENSE
└── .gitignore
```

## Installation

```
git clone https://github.com/DeadtommiVR/DeepFake-Detection-Temporal-Coherence.git
cd DeepFake-Detection-Temporal-Coherence
pip install -r requirements.txt
```

Open `TESSA_EDWARDS_DFD_V1.ipynb` (or a notebook under `notebooks/`) in Jupyter or Colab and run sequentially.

## Directions for Continuation

If this work were resumed with more time and compute:

- **3D spatiotemporal models** — 3D CNNs or video transformers (TimeSformer, Video Swin) for end-to-end spatiotemporal feature extraction rather than CNN + LSTM decoupling
- **Better optical flow** — replace Farneback with RAFT or PWC-Net for higher-fidelity flow estimation
- **Compression-aware evaluation** — explicitly train and evaluate on re-encoded versions of the dataset to measure the lab-to-deployment gap; this is the most important missing piece
- **Multi-scale feature fusion** — combine spatial, temporal, and frequency-domain features
- **AutoML / NAS** — automated architecture search rather than manual + Bayesian tuning

## Related Work

A full review of the detection literature — frame-based methods, the optical-flow and temporal-modelling lineage this project descends from, and the ensemble/generative state of the art — is in **[METHODOLOGY.md §2](METHODOLOGY.md)**, with citations.

At a higher level, this project also shares a conceptual thread with [Silent_Signal](https://github.com/DeadtommiVR/silent-signal) (forthcoming): both examine the gap between human perception and machine perception, and both confront the problem of *what signal survives lossy transmission* — through video compression in this case, through speaker playback and microphone capture in the audio case.

## Author

Tessa Edwards (DeadtommiVR) t@deadtommi.com

## License

MIT — see [LICENSE](https://github.com/DeadtommiVR/DeepFake-Detection-Temporal-Coherence/blob/main/LICENSE)