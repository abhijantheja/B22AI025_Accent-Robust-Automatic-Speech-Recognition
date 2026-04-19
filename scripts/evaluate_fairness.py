"""
Dedicated fairness evaluation script.
Generates ΔWERmax report, intersectional heatmaps, and demographic parity analysis.

Usage:
    python scripts/evaluate_fairness.py \
        --predictions_dir results/evaluation \
        --output_dir results/fairness_report
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evaluation.fairness import FairnessEvaluator
from src.evaluation.metrics import compute_delta_wer_max
from src.utils.helpers import get_logger


def parse_args():
    parser = argparse.ArgumentParser(description="Fairness Evaluation Report")
    parser.add_argument("--predictions_dir", type=str, default="results/evaluation")
    parser.add_argument("--output_dir", type=str, default="results/fairness_report")
    return parser.parse_args()


def load_all_predictions(predictions_dir: str) -> dict:
    """Load all prediction JSON files from evaluation directory."""
    results = {}
    for fname in os.listdir(predictions_dir):
        if fname.startswith("predictions_") and fname.endswith(".json"):
            path = os.path.join(predictions_dir, fname)
            with open(path) as f:
                data = json.load(f)
            results[data["model"]] = data
    return results


def plot_fairness_radar(per_accent_wers: dict, output_dir: str):
    """Radar chart comparing per-accent WER across models."""
    accents = sorted(set(
        acc for r in per_accent_wers.values() for acc in r.keys()
    ))
    N = len(accents)
    if N < 3:
        return

    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    colors = plt.cm.tab10(np.linspace(0, 1, len(per_accent_wers)))

    for (model, wer_dict), color in zip(per_accent_wers.items(), colors):
        values = [wer_dict.get(acc, 0) * 100 for acc in accents]
        values += values[:1]
        ax.plot(angles, values, "o-", linewidth=2, label=model, color=color)
        ax.fill(angles, values, alpha=0.1, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(accents, fontsize=10)
    ax.set_title("Per-Accent WER Radar Chart", fontsize=14, pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    ax.grid(True)

    path = os.path.join(output_dir, "fairness_radar.png")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Radar chart saved: {path}")


def generate_fairness_summary(all_results: dict, output_dir: str):
    """Generate LaTeX-ready fairness table."""
    rows = []
    for model, data in all_results.items():
        metrics = data.get("metrics", {})
        per_accent = metrics.get("per_accent_wer", {})
        rows.append({
            "Model": model,
            "Overall WER (%)": round(metrics.get("wer", 0) * 100, 2),
            "ΔWERmax (%)": round(metrics.get("delta_wer_max", 0) * 100, 2),
            "Worst Accent WER (%)": round(max(per_accent.values(), default=0) * 100, 2),
            "Best Accent WER (%)": round(min(per_accent.values(), default=0) * 100, 2),
        })

    df = pd.DataFrame(rows)
    print("\nFairness Summary:")
    print(df.to_string(index=False))

    csv_path = os.path.join(output_dir, "fairness_summary.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")

    # LaTeX table
    latex_path = os.path.join(output_dir, "fairness_table.tex")
    with open(latex_path, "w") as f:
        f.write("% Fairness Evaluation Table (auto-generated)\n")
        f.write("\\begin{table}[h]\n\\centering\n")
        f.write("\\begin{tabular}{lcccc}\n\\hline\n")
        f.write("Model & Overall WER (\\%) & $\\Delta$WERmax (\\%) & Worst Accent (\\%) & Best Accent (\\%) \\\\\n\\hline\n")
        for _, row in df.iterrows():
            f.write(f"{row['Model']} & {row['Overall WER (%)']} & {row['ΔWERmax (%)']} & "
                    f"{row['Worst Accent WER (%)']} & {row['Best Accent WER (%)']} \\\\\n")
        f.write("\\hline\n\\end{tabular}\n")
        f.write("\\caption{Fairness evaluation across accent groups. Lower $\\Delta$WERmax indicates more equitable performance.}\n")
        f.write("\\label{tab:fairness}\n\\end{table}\n")
    print(f"LaTeX table saved: {latex_path}")
    return df


def main():
    args = parse_args()
    logger = get_logger("evaluate_fairness")
    os.makedirs(args.output_dir, exist_ok=True)

    all_results = load_all_predictions(args.predictions_dir)

    if not all_results:
        logger.error(f"No prediction files found in {args.predictions_dir}")
        logger.info("Run evaluate.py first to generate predictions.")
        return

    logger.info(f"Found {len(all_results)} model results: {list(all_results.keys())}")

    # Per-accent WER for radar
    per_accent_wers = {}
    for model, data in all_results.items():
        metrics = data.get("metrics", {})
        if "per_accent_wer" in metrics:
            per_accent_wers[model] = metrics["per_accent_wer"]

    if per_accent_wers:
        plot_fairness_radar(per_accent_wers, args.output_dir)

    fairness_eval = FairnessEvaluator(output_dir=args.output_dir)
    if per_accent_wers:
        fairness_eval.plot_per_accent_wer(
            {m: {"per_accent_wer": wers} for m, wers in per_accent_wers.items()},
            save_path=os.path.join(args.output_dir, "per_accent_bar.png"),
        )

    df = generate_fairness_summary(all_results, args.output_dir)
    logger.info(f"Fairness report complete. Output: {args.output_dir}")


if __name__ == "__main__":
    main()
