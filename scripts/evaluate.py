"""
Phase 3: Comprehensive evaluation on multiple accent test sets.
Reports WER, CER, ΔWERmax per model and generates comparison tables.

Usage:
    python scripts/evaluate.py \
        --checkpoints results/phase0_baseline/best_model.pt \
                      results/phase2_adversarial/best_model.pt \
        --test_manifest data/manifests/test_accented.jsonl \
        --output_dir results/evaluation
"""

import os
import sys
import json
import argparse
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.adversarial_asr import AdversarialASRModel
from src.data.dataset import create_dataloader, AccentedSpeechDataset, collate_fn
from src.data.preprocessing import AudioPreprocessor
from src.evaluation.metrics import compute_all_metrics, print_metrics_table, compute_wer
from src.evaluation.fairness import FairnessEvaluator
from src.utils.helpers import set_seed, get_logger, load_checkpoint
from torch.utils.data import DataLoader
from transformers import Wav2Vec2Processor
import yaml


MODEL_NAMES = {
    "phase0": "Baseline (LibriSpeech)",
    "phase1": "Accent-Adapted",
    "phase2": "Adversarial (Ours)",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Phase 3: Comprehensive Evaluation")
    parser.add_argument("--checkpoints", nargs="+", required=True,
                        help="Paths to model checkpoints to evaluate")
    parser.add_argument("--model_labels", nargs="+", default=None,
                        help="Labels for each checkpoint (same order)")
    parser.add_argument("--test_manifest", type=str, required=True)
    parser.add_argument("--config", type=str, default="configs/adversarial_config.yaml")
    parser.add_argument("--output_dir", type=str, default="results/evaluation")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


@torch.no_grad()
def run_inference(model, loader, processor, device) -> dict:
    """Run inference and collect all predictions, references, and metadata."""
    model.eval()
    all_refs, all_hyps, all_accents, all_genders, all_ages = [], [], [], [], []

    for batch in loader:
        input_values = batch["input_values"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        ctc_logits, _, _ = model(
            input_values=input_values,
            attention_mask=attention_mask,
            use_adversarial=False,
        )

        # Greedy CTC decode
        pred_ids = ctc_logits.argmax(dim=-1)
        predictions = processor.batch_decode(pred_ids)

        all_refs.extend(batch["texts"])
        all_hyps.extend(predictions)
        all_accents.extend(batch["accents"])
        all_genders.extend(batch["genders"])
        all_ages.extend(batch["ages"])

    return {
        "references": all_refs,
        "hypotheses": all_hyps,
        "accents": all_accents,
        "genders": all_genders,
        "ages": all_ages,
    }


def plot_wer_comparison(all_results: dict, output_dir: str):
    """Side-by-side bar chart of overall WER and ΔWERmax."""
    models = list(all_results.keys())
    overall_wers = [all_results[m]["wer"] * 100 for m in models]
    delta_wers = [all_results[m].get("delta_wer_max", 0) * 100 for m in models]

    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width / 2, overall_wers, width, label="Overall WER (%)", color="steelblue")
    bars2 = ax.bar(x + width / 2, delta_wers, width, label="ΔWERmax (%)", color="tomato")

    ax.set_xlabel("Model", fontsize=12)
    ax.set_ylabel("WER (%)", fontsize=12)
    ax.set_title("ASR Performance & Fairness Comparison", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{bar.get_height():.1f}%", ha="center", va="bottom", fontsize=9)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{bar.get_height():.1f}%", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    path = os.path.join(output_dir, "wer_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Comparison plot saved: {path}")


def main():
    args = parse_args()
    logger = get_logger("evaluate")
    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Evaluating {len(args.checkpoints)} models on {args.test_manifest}")

    preprocessor = AudioPreprocessor(processor_name=config.get("model_name", "facebook/wav2vec2-base"))
    processor = preprocessor.processor

    test_dataset = AccentedSpeechDataset(
        manifest_path=args.test_manifest,
        preprocessor=preprocessor,
        split="test",
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    labels = args.model_labels or [f"Model {i+1}" for i in range(len(args.checkpoints))]
    fairness_eval = FairnessEvaluator(output_dir=os.path.join(args.output_dir, "fairness"))

    all_results = {}
    all_fairness = {}

    for ckpt_path, label in zip(args.checkpoints, labels):
        logger.info(f"\n--- Evaluating: {label} ---")

        model = AdversarialASRModel(
            model_name=config.get("model_name", "facebook/wav2vec2-base"),
            vocab_size=config.get("vocab_size", 32),
            num_accents=config.get("num_accents", 8),
        )
        model, _, _ = load_checkpoint(ckpt_path, model, device=device)
        model = model.to(device)

        outputs = run_inference(model, test_loader, processor, device)

        metrics = compute_all_metrics(
            references=outputs["references"],
            hypotheses=outputs["hypotheses"],
            accents=outputs["accents"],
            genders=outputs["genders"],
            ages=outputs["ages"],
        )
        all_results[label] = metrics
        print_metrics_table(metrics)

        # Fairness analysis
        fairness = fairness_eval.evaluate(
            references=outputs["references"],
            hypotheses=outputs["hypotheses"],
            accents=outputs["accents"],
            genders=outputs["genders"],
            ages=outputs["ages"],
            model_name=label,
        )
        all_fairness[label] = fairness

        # Save per-model predictions
        pred_path = os.path.join(args.output_dir, f"predictions_{label.replace(' ', '_')}.json")
        with open(pred_path, "w") as f:
            json.dump({
                "model": label,
                "checkpoint": ckpt_path,
                "metrics": metrics,
                "samples": [
                    {"ref": r, "hyp": h, "accent": a}
                    for r, h, a in zip(
                        outputs["references"][:100],
                        outputs["hypotheses"][:100],
                        outputs["accents"][:100],
                    )
                ],
            }, f, indent=2)

    # Comparison plots
    plot_wer_comparison(all_results, args.output_dir)
    fairness_eval.plot_per_accent_wer(all_fairness, save_path=os.path.join(args.output_dir, "per_accent_comparison.png"))

    # Summary table
    summary = []
    for label, metrics in all_results.items():
        row = {
            "Model": label,
            "WER (%)": f"{metrics['wer']*100:.2f}",
            "CER (%)": f"{metrics['cer']*100:.2f}",
            "ΔWERmax (%)": f"{metrics.get('delta_wer_max', 0)*100:.2f}",
        }
        if "per_accent_wer" in metrics:
            for acc, wer in metrics["per_accent_wer"].items():
                row[f"WER_{acc} (%)"] = f"{wer*100:.2f}"
        summary.append(row)

    df = pd.DataFrame(summary)
    csv_path = os.path.join(args.output_dir, "summary_table.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nSummary table saved: {csv_path}")
    print(df.to_string(index=False))

    fairness_eval.save_report(all_fairness)
    logger.info(f"\nAll results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
