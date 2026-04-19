#!/usr/bin/env bash
# End-to-end reproduction script.
# Run from project root: bash scripts/run_all.sh
# Assumes data is already prepared in data/manifests/

set -e

echo "============================================="
echo "  Accent-Robust ASR — Full Experiment Run"
echo "  B22AI023 / B22AI005 / B22AI025"
echo "============================================="

SEED=42

# Phase 0: Baseline
echo ""
echo "[Phase 0] Baseline training on LibriSpeech..."
python scripts/train_baseline.py \
    --config configs/baseline_config.yaml \
    --output_dir results/phase0_baseline \
    --seed $SEED

# Phase 1: Accent adaptation
echo ""
echo "[Phase 1] Accent adaptation with stratified sampling..."
python scripts/train_accent_adaptation.py \
    --config configs/adversarial_config.yaml \
    --pretrained results/phase0_baseline/best_model.pt \
    --output_dir results/phase1_adaptation \
    --seed $SEED

# Phase 2: Adversarial training
echo ""
echo "[Phase 2] Domain-Adversarial Training (GRL)..."
python scripts/train_adversarial.py \
    --config configs/adversarial_config.yaml \
    --pretrained results/phase1_adaptation/best_model.pt \
    --output_dir results/phase2_adversarial \
    --seed $SEED

# Phase 3: Evaluation
echo ""
echo "[Phase 3] Comprehensive evaluation..."
python scripts/evaluate.py \
    --checkpoints \
        results/phase0_baseline/best_model.pt \
        results/phase1_adaptation/best_model.pt \
        results/phase2_adversarial/best_model.pt \
    --model_labels "Baseline" "Accent-Adapted" "Adversarial (Ours)" \
    --test_manifest data/manifests/accented_test.jsonl \
    --output_dir results/evaluation \
    --seed $SEED

# Fairness report
echo ""
echo "[Fairness] Generating fairness analysis..."
python scripts/evaluate_fairness.py \
    --predictions_dir results/evaluation \
    --output_dir results/fairness_report

echo ""
echo "============================================="
echo "  All experiments complete!"
echo "  Results: results/"
echo "  Summary: results/evaluation/summary_table.csv"
echo "  Fairness: results/fairness_report/"
echo "============================================="
