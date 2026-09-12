# FedSeMER: Federated Semi-Supervised Multimodal Emotion Recognition

> Official code repository for the paper **"Federated Semi-Supervised Learning for Multimodal Emotion Recognition under Heterogeneous Client Distributions"** (preprint submitted to Elsevier).

<div align="center">

[![python](https://img.shields.io/badge/-Python_3.8.20-blue?logo=python&logoColor=white)](https://www.python.org/downloads/)
[![pytorch](https://img.shields.io/badge/Torch_2.0.1-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![cuda](https://img.shields.io/badge/-CUDA_11.8-green?logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit-archive)

</div>

<p align="center">
<img src="https://img.shields.io/badge/Last%20updated%20on-04.09.2026-brightgreen?style=for-the-badge">
<img src="https://img.shields.io/badge/Written%20by-Nguyen%20Minh%20Nhut-pink?style=for-the-badge">
</p>

<div align="center">

[**Overview**](#overview) •
[**Highlights**](#highlights) •
[**Framework**](#framework) •
[**Repository Structure**](#repository-structure) •
[**Setup**](#setup) •
[**Quick Start**](#quick-start) •
[**LOSO Runners**](#loso-runners) •
[**Results**](#results) •
[**Datasets**](#datasets) •
[**References**](#references) •
[**Citation**](#citation) •
[**Contact**](#contact)

</div>

## Overview

Multimodal Emotion Recognition (MER) infers a speaker's emotional state by jointly exploiting speech and linguistic content. In practice, emotion data is collected from highly private conversations, and labeling is costly and time-consuming, so real deployments must cope with **data that cannot leave its source** and with **mostly unlabeled data**.

**FedSeMER** is a federated semi-supervised framework for MER that jointly addresses:
- **Higher-order dependency modeling** among utterances (beyond pairwise attention),
- **Label scarcity**, by exploiting large amounts of unlabeled conversational data, and
- **Data-local, decentralized training**, where raw audio and text never leave the client and only model parameters are exchanged with the server.

FedSeMER is positioned as **reducing the direct exposure of raw conversational data**, not as providing a formal privacy guarantee — shared model updates may still carry recoverable information about local data.

## Highlights

- **Dual-stream client architecture**: [Dynamic Hypergraph Learning (DHL)](#framework) captures higher-order intra-/inter-modal relationships between utterances; a [Gated Cross-Modal Transformer (GCMT)](#framework) models reliability-aware bidirectional audio↔text interaction; a [Gated Multimodal Unit (GMU)](#framework) adaptively fuses both branches.
- **Two-stage federated optimization**: Stage I performs federated supervised initialization; Stage II performs FixMatch-style confidence-thresholded pseudo-labeling for semi-supervised refinement.
- **Dynamic pseudo-label-aware aggregation**: each client's contribution to the global update is weighted by its labeled samples plus its confidently pseudo-labeled samples, and this weight evolves across communication rounds.
- **FedAvg and FedProx** are both supported as server-side aggregation strategies.
- **Dataset-specific, disjoint client partitioning**: speaker-disjoint clients (LOSO-style) for IEMOCAP/MSP-IMPROV, and dialogue-disjoint clients under official splits for MELD.
- Evaluated across multiple **labeled-data ratios** (0.3 / 0.5 / 0.7 / 1.0) and **client counts** (3 / 5 / 7), against a centralized-learning reference and four re-implemented MER baselines (CemoBAM, MemoCMT, 3M-SER, FleSER).

## Framework

FedSeMER is described from two complementary perspectives:

**(a) End-to-end local learning pipeline** (per client):
1. **Supervised pre-training** — audio is encoded with **Wav2Vec 2.0** and text with **BERT**; the client model is trained on the labeled subset with class-weighted cross-entropy.
2. **FixMatch-style pseudo-label generation** — for unlabeled samples, a weakly augmented view (small word-dropout / small Gaussian noise) produces a pseudo-label and confidence score; a strongly augmented view (larger word-dropout / larger noise) is used for the consistency loss. Only samples with confidence above threshold `τ` are accepted.
3. **Semi-supervised local optimization** — the client jointly minimizes the supervised loss and a confidence-weighted pseudo-label loss.

**(b) Client-side multimodal model** (dual-stream, fused by GMU):
- **Dynamic Hypergraph Learning (DHL)** — builds a heterogeneous hypergraph over utterance nodes with intra-modal hyperedges (global context per modality) and Top-K similarity-based inter-modal hyperedges, then applies an R-GCN followed by a Graph Transformer to obtain structure-aware representations.
- **Gated Cross-Modal Transformer (GCMT)** — applies bidirectional cross-attention (audio→text and text→audio) and learns a sigmoid gate per direction so that only the useful portion of cross-modal context is injected back into each modality stream, reducing negative transfer from a noisy/unreliable modality.
- **Gated Multimodal Unit (GMU)** — a feature-wise sigmoid gate `z` adaptively combines `tanh(F_DHL)` and `tanh(F_GCMT)` into the final fused representation used for emotion classification.

**(c) Federated workflow**:
- A central server coordinates `K` clients, each holding a disjoint labeled subset `D_L^k` and unlabeled subset `D_U^k`.
- **Stage I (Rounds 1…R₁, federated supervised initialization)**: each client trains locally on `D_L^k` for `E` epochs (optionally with the FedProx proximal term), and the server aggregates by sample-size-weighted averaging (`n_k = |D_L^k|`).
- **Stage II (Rounds R₁+1…R₁+R₂, federated semi-supervised refinement)**: the previous global model generates confidence-filtered pseudo-labels on `D_U^k`; the aggregation weight becomes `n_k = |D_L^k| + |confidently-pseudo-labeled subset|`, so it can change dynamically round-to-round as pseudo-label yield changes.
- Only trainable model parameters are exchanged each round — raw audio and text never leave the client.

## Repository Structure

```text
centralized/        # centralized training, MELD pipeline, LOSO runner
federated/          # federated preprocess/train/eval pipelines
feature_extract/    # feature extraction CLI
src/                # model architectures + shared utilities
demo_FL/            # FastAPI + Streamlit demo for global/local FL simulation
metadata/           # generated metadata PKLs
features/           # extracted feature PKLs
checkpoints/        # saved model checkpoints
logs/               # training/evaluation logs
NoiseX-92/           # example noise files used for noisy-test generation
```

## Setup

Use Python 3.10+ (the Dockerfiles use `python:3.10-slim`).

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Recommended extras used by scripts/UI:

```bash
pip install librosa streamlit
```

## Quick Start

Typical data processing workflow in this repository:

### 1) Preprocess metadata

```bash
# centralized metadata
python3 centralized/preprocess.py \
  --dataset MELD \
  --data_root /path/to/data_root \
  --out_dir metadata/MELD_preprocessed

# federated client splits
python3 federated/preprocess.py \
  --dataset MELD \
  --data_root /path/to/data_root \
  --out_dir metadata/MELD_federated/clients \
  --split_by speaker \
  --num_clients 5 \
  --labeled_ratio 1.0 \
  --seed 42
```

### 2) Extract features

```bash
# centralized features
python3 feature_extract/extract_feature.py \
  --dataset MELD \
  --pkl_dir metadata/MELD_preprocessed \
  --output_dir features/MELD/centralized \
  --wav_base /path/to/data_root

# federated features (per client)
for c in metadata/MELD_federated/clients/client_*; do
  python3 feature_extract/extract_feature.py \
    --dataset MELD \
    --client_dir "$c" \
    --out_dir "features/MELD/clients/$(basename "$c")" \
    --wav_base /path/to/data_root
done
```

### 3) Train model

```bash
# centralized training
python3 centralized/train.py \
  --data_dir features/MELD/centralized \
  --dataset MELD \
  --num_classes 7 \
  --model_name fedsemer \
  --epochs 50

# federated training
python3 -m federated.run_federated \
  --dataset MELD \
  --num_classes 7 \
  --model_name fedsemer \
  --clients_root metadata/MELD_federated/clients \
  --features_root features/MELD/clients \
  --rounds_pretrain 10 \
  --rounds_ssl 0
```

Key hyperparameters used in the paper (see `federated/configs/*.yaml` to reproduce): AdamW, initial LR `1e-3` with a scheduler, batch size `32`, `E=5` local epochs/round, FedProx coefficient `μ=0.01` (`μ=0` = FedAvg), `R1=30` supervised rounds + `R2=20` semi-supervised rounds (50 total), DHL Top-`K=5` cross-modal neighbors, pseudo-label confidence threshold `τ=0.96`, unlabeled-loss weight `λ_u=0.3`.

### 4) Run full pipeline (shortcut)

```bash
# centralized all-in-one
python3 centralized/run_meld_pipeline.py --config centralized/configs/meld.yaml

# federated all-in-one
python3 federated/run_meld_pipeline.py --config federated/configs/meld.yaml
```

## LOSO Runners

### Centralized LOSO

```bash
python3 centralized/loso_runner.py --config centralized/configs/iemocap.yaml
```

### Federated LOSO

```bash
python3 federated/loso_runner.py --config federated/configs/iemocap.yaml
```

Both runners use session-based LOSO folds and output per-fold/per-seed summaries. IEMOCAP and MSP-IMPROV use leave-one-speaker-out, speaker-disjoint client partitioning; MELD keeps the official dialogue-disjoint train/val/test split with one client per training dialogue.

## Outputs

- `metadata/...`: preprocessed train/val/test PKLs, client partitions, `client_map.json`
- `features/...`: extracted embeddings (`*_features.pkl`, optional manifests)
- `checkpoints/...`: centralized/federated checkpoints
- `logs/...`: run logs, per-round/client metrics, eval summaries
- `results/...`: aggregated centralized metrics CSV

## Results

Best FedSeMER results at labeled ratio = 1.0 (mean over 3 seeds), against the centralized-learning reference:

| Dataset | Best federated setting | WA (%) | UA (%) | Centralized WA (%) | Centralized UA (%) |
|---|---|---|---|---|---|
| IEMOCAP | 5 clients, FedProx | 71.23 | 73.17 | 72.43 | 73.92 |
| MSP-IMPROV | 3 clients, FedAvg | 48.16 | 48.53 | 49.78 | 49.87 |
| MELD | 7 clients, FedAvg | 55.83 | 44.50 | 55.29 | 46.50 |

Notable ablation findings (full tables in the paper):
- **Modules**: removing DHL causes the largest drop (e.g., IEMOCAP WA 71.23→69.43, UA 73.17→70.71 at ratio 1.0), followed by GCMT and GMU — all three components contribute positively, and parallel dual-stream + GMU fusion outperforms sequential DHL→GCMT / GCMT→DHL variants.
- **Modality**: multimodal fusion is always best; text alone beats audio alone by a wide margin on all three datasets (largest gap on MELD, narrowest on MSP-IMPROV).
- **Server-side aggregation**: federated aggregation (FedAvg/FedProx) substantially outperforms local-only (no-aggregation) training, with the gap widening as the number of clients grows.
- **Dynamic pseudo-label-aware weighting** outperforms both static (labeled-count-only) and confidence-mass weighting on IEMOCAP and (mostly) on MELD.
- **Confidence threshold** `τ=0.96` gives the best quantity/quality trade-off on IEMOCAP; `τ=0.98` starts discarding too many usable pseudo-labels.
- **Robustness**: under Babble / F-16 / HF-Channel noise (NoiseX-92) injected into the audio modality at SNR 10/5/0, GMU fusion degrades more gracefully than Concat, `[CLS]`, self-attention, or cross-attention fusion.

## Datasets

- **[IEMOCAP](https://sail.usc.edu/iemocap/)** — acted, dyadic, multimodal corpus; 4-class setup (angry, happy, sad, neutral; `excited` merged into `happy`). Evaluated with leave-one-speaker-out, 10-fold cross-validation; clients are speaker-disjoint.
- **[MSP-IMPROV](https://ecs.utdallas.edu/research/researchlabs/msp-lab/MSP-Improv.html)** — acted, dyadic, spontaneous-improvisation corpus; 6 sessions / 12 actors, 8,438 turns; 4 primary emotions (angry, happy, sad, neutral); speaker-disjoint clients.
- **[MELD](https://affective-meld.github.io/)** — multi-party conversational corpus from *Friends*; 1,400+ dialogues, ~13,000 utterances, 7 emotion labels. Official dialogue-disjoint train/val/test splits are retained; one client per training dialogue (speaker overlap across clients is allowed, dialogue overlap is not).

## Limitations

- Limited to **audio + text**; visual and physiological modalities are not yet integrated.
- The federated setting is evaluated **in simulation** and does not model unreliable communication, client dropout, or heterogeneous hardware.
- FedSeMER **reduces direct data exposure** through data-local training and disjoint partitioning; it does **not** provide a formal privacy guarantee, and no leakage/differential-privacy evaluation is included in this version.
- Performance on MELD is less stable across labeled ratios/client configurations than on IEMOCAP/MSP-IMPROV, due to class imbalance and heterogeneous multi-party client distributions.

## License

This project is licensed under the MIT License. See `LICENSE`.

## References
[1] Nhut Minh Nguyen, Enhancing multimodal emotion recognition with dynamic fuzzy membership and attention fusion, (Engineering Applications of Artificial Intelligence), 2026. Available https://github.com/aita-lab/FleSER.

[2] Nhut Minh Nguyen, CemoBAM: Advancing Multimodal Emotion Recognition through Heterogeneous Graph Networks and Cross-Modal Attention Mechanisms (APNOMS), 2025. Available https://github.com/nhut-ngnn/CemoBAM.

[3] Nhat Truong Pham, SERVER: Multi-modal Speech Emotion Recognition using Transformer-based and Vision-based Embeddings (ICIIT), 2023. Available https://github.com/nhattruongpham/mmser.git.

[4] Mustaqeem Khan, MemoCMT: Cross-Modal Transformer-Based Multimodal Emotion Recognition System (Scientific Reports), 2025. Available https://github.com/tpnam0901/MemoCMT.

[5] Nhat Truong Pham, SER-Fuse: An Emotion Recognition Application Utilizing Multi-Modal, Multi-Lingual, and Multi-Feature Fusion (SOICT), 2023. Available https://github.com/nhattruongpham/SER-Fuse.

[6] Nhut Minh Nguyen, Nhat Truong Pham, Duc Ngoc Minh Dang, HyperDyG: Hypergraph-driven dynamic fusion for semi-supervised multimodal emotion recognition (EAI Endorsed Transactions on Industrial Networks and Intelligent Systems), 2026.

[7] Nhut Minh Nguyen, Nhat Truong Pham, Duc Ngoc Minh Dang, SemiFedER: Semi-supervised federated averaging for multimodal emotion recognition (MLHMI), 2026.

## Citation

If you use this repository or build on FedSeMER, please cite the paper:

```bibtex
@article{nguyen2026fedsemer,
  title   = {Federated Semi-Supervised Learning for Multimodal Emotion Recognition under Heterogeneous Client Distributions},
  author  = {Nguyen, Nhut Minh and Pham, Nhat Truong and Tran, Phuong-Nam and Le, Linh and Othmani, Alice and Lim, Chee Peng and Dang, Duc Ngoc Minh},
  journal = {Preprint submitted to Elsevier},
  year    = {2026}
}
```

## Contact

- Email: `minhnhut.ngnn@gmail.com`
- GitHub: https://github.com/nhut-ngnn
- ORCID: https://orcid.org/0009-0003-1281-5346

