# Accent-Robust Automatic Speech Recognition using Wav2Vec 2.0 with Adversarial Training

**Course**: Speech & Audio Processing  
**Team**: Karan Reddy (B22AI023) · Ale Anwesh (B22AI005) · Abhijan Theja (B22AI025)

---

## Overview

This repository implements an accent-robust Automatic Speech Recognition (ASR) system that combines **Wav2Vec 2.0** self-supervised representations with **Domain-Adversarial Training (DAT)**. A Gradient Reversal Layer (GRL) forces the encoder to learn accent-invariant phonetic features, improving transcription accuracy for non-native and underrepresented accent groups.

### Key Contributions
- Wav2Vec 2.0 + GRL adversarial training framework for accent-invariant ASR
- Accent-stratified sampling to prevent majority-accent dominance during training
- Comprehensive **Responsible AI** analysis: ΔWERmax fairness metric, intersectional analysis (accent × gender × age), and Grad-CAM explainability

---

## Architecture

```
Raw Audio (16kHz)
      │
      ▼
┌─────────────────────┐
│  Wav2Vec 2.0 Encoder │   (frozen CNN feature extractor → Transformer)
└─────────────────────┘
      │  Hidden States (B, T', 768)
      ├──────────────────────────────────────┐
      ▼                                      ▼
┌──────────────┐                   ┌──────────────────────────┐
│  CTC Decoder  │                   │    Accent Classifier      │
│  (LM Head)    │                   │  ┌────────────────────┐  │
└──────────────┘                   │  │  Supervised Head    │  │
      │                            │  └────────────────────┘  │
      ▼                            │  ┌────────────────────┐  │
  Transcription                    │  │  GRL → Adversarial  │  │
                                   │  └────────────────────┘  │
                                   └──────────────────────────┘
```

**Training Loss** = L_CTC + λ_sup · L_accent_sup + λ_adv · L_accent_adv  
*(GRL negates gradients for the adversarial head, making the encoder accent-invariant)*

---

## Project Structure

```
project/
├── src/
│   ├── models/
│   │   ├── gradient_reversal.py      # GRL module with annealed alpha
│   │   ├── accent_classifier.py      # Supervised + adversarial accent heads
│   │   └── adversarial_asr.py        # Main model: Wav2Vec2 + CTC + GRL
│   ├── data/
│   │   ├── dataset.py                # AccentedSpeechDataset + collate_fn
│   │   ├── preprocessing.py          # Audio loading, resampling, normalization
│   │   └── sampling.py               # Accent-stratified sampler
│   ├── training/
│   │   ├── trainer.py                # Full training loop with AMP
│   │   └── losses.py                 # CTC + accent classification losses
│   ├── evaluation/
│   │   ├── metrics.py                # WER, CER, ΔWERmax
│   │   ├── fairness.py               # Fairness analysis & plots
│   │   └── explainability.py         # Grad-CAM visualization
│   └── utils/
│       └── helpers.py                # Seeding, checkpointing, logging
├── scripts/
│   ├── prepare_data.py               # Dataset download & manifest creation
│   ├── train_baseline.py             # Phase 0: LibriSpeech CTC baseline
│   ├── train_accent_adaptation.py    # Phase 1: Accent fine-tuning
│   ├── train_adversarial.py          # Phase 2: DAT with GRL
│   ├── evaluate.py                   # Phase 3: WER + fairness evaluation
│   ├── evaluate_fairness.py          # Detailed fairness report + LaTeX table
│   └── visualize_gradcam.py          # Grad-CAM explainability
├── configs/
│   ├── baseline_config.yaml          # Phase 0 hyperparameters
│   └── adversarial_config.yaml       # Phase 1/2 hyperparameters
├── results/                          # Experiment outputs
├── report/
│   └── report.tex                    # CVPR-format report
├── README.md
└── requirements.txt
```

---

## Installation

### Prerequisites
- Python >= 3.9
- GPU recommended (CPU works but slower). CUDA 11.8 or 12.x.

### Setup

```bash
# 1. Clone the repository
git clone https://github.com/abhijantheja/Accent-Robust-Automatic-Speech-Recognition.git
cd Accent-Robust-Automatic-Speech-Recognition

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate         # Linux/Mac
# venv\Scripts\activate          # Windows

# 3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 4. Smoke test (no data needed)
python scripts/demo.py
```

---

## Running Experiments

### Option A — Full run (all 3 phases, ~4–6 hours on GPU)

```bash
python scripts/quick_train_eval.py \
    --output_dir results/ \
    --phase1_epochs 3 \
    --phase2_epochs 5 \
    --batch_size 4 \
    --seed 42
```

### Option B — Baseline evaluation only (~20–30 minutes)

```bash
python scripts/quick_train_eval.py \
    --output_dir results/ \
    --phase 0_eval_only \
    --seed 42
```


### Outputs

| File | Description |
|------|-------------|
| `results/evaluation/all_results.json` | All WER/ΔWERmax numbers |
| `results/evaluation/wer_comparison.png` | Overall WER bar chart |
| `results/evaluation/per_accent_wer.png` | Per-accent grouped bar chart |
| `results/evaluation/heatmap.png` | WER heatmap across models & accents |
| `results/evaluation/training_curves.png` | Loss + WER training curves |
| `results/evaluation/results_table.tex` | LaTeX results table |
| `results/phase2_adversarial/phase2_best.pt` | Best adversarial model checkpoint |

---

## Full Pipeline with Local Datasets

If you have the datasets downloaded locally:

### Datasets

| Dataset | Accents | Size | Purpose |
|---------|---------|------|---------|
| LibriSpeech (train-clean-100) | American English | ~6 GB | Phase 0 baseline |
| CommonVoice (English, v13) | 8+ accent groups | ~65 GB | Phases 1 & 2 |
| L2-ARCTIC | Arabic, Chinese, Indian, Korean, Spanish, Vietnamese | ~3 GB | Evaluation |
| AESRC2020 | 8 accent groups | ~30 GB | Additional training |

```bash
# Build manifests from local data
python scripts/prepare_data.py \
    --dataset all \
    --librispeech_dir /path/to/LibriSpeech \
    --cv_dir /path/to/cv-corpus-13.0/en \
    --l2arctic_dir /path/to/l2arctic

# Run all phases
bash scripts/run_all.sh
```

---

## Results

Evaluated on LibriSpeech, CommonVoice, and L2-ARCTIC across 8 accent groups with seed 42.

| Model | Overall WER (%) | ΔWERmax (%) |
|-------|----------------|------------|
| Baseline (`wav2vec2-base-960h`) | 27.96 | 24.73 |
| Accent-Adapted (Phase 1) | 20.33 | 19.04 |
| **Adversarial (Ours)** | **18.15** | **16.10** |

Our adversarial model achieves a **35.1% relative WER reduction** and a **34.9% reduction in ΔWERmax** compared to the baseline, demonstrating improved performance and fairness across all accent groups.

---

## Responsible AI

### Fairness
- **Primary metric**: ΔWERmax = max(WER_i) − min(WER_i) across accent groups
- Evaluated across 8 accent groups from CommonVoice and L2-ARCTIC

### Inclusiveness
- Intersectional analysis: WER broken down by (accent × gender) and (accent × gender × age)
- Accent-stratified training sampling to prevent majority dominance

### Explainability
- Grad-CAM visualizations over Wav2Vec 2.0 encoder layers
- Highlights temporal speech regions driving correct/incorrect predictions

### Robustness
- Balanced accent-stratified sampling during all fine-tuning phases
- Evaluation on unseen accent groups (zero-shot transfer)

---

## Reproducibility Checklist

- [x] Random seed set to 42 in all scripts (`--seed 42`)
- [x] All hyperparameters documented in `configs/`
- [x] Training history saved as JSON (`training_history.json`)
- [x] Model checkpoints saved per epoch and at best WER
- [x] Evaluation outputs saved as CSV + JSON for exact reproduction

---

## Citation

```bibtex
@misc{reddy2024accentasr,
  title={Accent-Robust ASR using Wav2Vec 2.0 with Adversarial Training},
  author={Reddy, Karan and Anwesh, Ale and Theja, Abhijan},
  year={2024},
  institution={IIT Jodhpur}
}
```

## References

1. A. Baevski et al., "wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations," NeurIPS, 2020.
2. Y. Ganin et al., "Domain-Adversarial Training of Neural Networks," JMLR, 2016.
3. J. Li et al., "Accent-Robust ASR using Supervised and Unsupervised Wav2Vec Embeddings," arXiv, 2021.
4. G. Zhao et al., "L2-ARCTIC: A Non-Native English Speech Corpus," Interspeech, 2018.
5. R. Ardila et al., "Common Voice: A Massively-Multilingual Speech Corpus," LREC, 2020.
