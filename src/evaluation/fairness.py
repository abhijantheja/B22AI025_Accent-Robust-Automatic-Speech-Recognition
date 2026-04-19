"""
Fairness evaluation for accent-robust ASR.
Implements ΔWERmax, intersectional analysis, and demographic parity checks.
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

from .metrics import compute_wer, compute_per_accent_wer, compute_delta_wer_max


class FairnessEvaluator:
    """
    Comprehensive fairness analysis for ASR systems.

    Metrics:
    - ΔWERmax: Maximum WER disparity across accent groups
    - Demographic parity: Equal error rates across protected groups
    - Intersectional fairness: WER across accent × gender × age combinations
    """

    def __init__(self, output_dir: str = "results/fairness"):
        self.output_dir = output_dir
        import os
        os.makedirs(output_dir, exist_ok=True)

    def evaluate(
        self,
        references: List[str],
        hypotheses: List[str],
        accents: List[str],
        genders: Optional[List[str]] = None,
        ages: Optional[List[str]] = None,
        model_name: str = "model",
    ) -> Dict:
        """
        Full fairness evaluation pipeline.

        Returns dict with all fairness metrics.
        """
        results = {
            "model": model_name,
            "overall_wer": compute_wer(references, hypotheses),
            "per_accent_wer": compute_per_accent_wer(references, hypotheses, accents),
        }
        results["delta_wer_max"] = compute_delta_wer_max(results["per_accent_wer"])

        # Gender-based analysis
        if genders:
            results["per_gender_wer"] = self._group_wer(references, hypotheses, genders)
            results["gender_delta_wer"] = compute_delta_wer_max(results["per_gender_wer"])

        # Age-based analysis
        if ages:
            results["per_age_wer"] = self._group_wer(references, hypotheses, ages)

        # Intersectional analysis
        if genders:
            results["intersectional_accent_gender"] = self._intersectional_wer(
                references, hypotheses, accents, genders
            )

        if ages and genders:
            results["intersectional_accent_age_gender"] = self._intersectional_wer_3way(
                references, hypotheses, accents, genders, ages
            )

        # Worst-group accuracy (WGA)
        results["worst_group_wer"] = max(results["per_accent_wer"].values())
        results["best_group_wer"] = min(results["per_accent_wer"].values())

        return results

    def _group_wer(
        self,
        refs: List[str],
        hyps: List[str],
        groups: List[str],
    ) -> Dict[str, float]:
        group_refs = defaultdict(list)
        group_hyps = defaultdict(list)
        for r, h, g in zip(refs, hyps, groups):
            group_refs[g].append(r)
            group_hyps[g].append(h)
        return {g: compute_wer(group_refs[g], group_hyps[g]) for g in group_refs}

    def _intersectional_wer(
        self,
        refs: List[str],
        hyps: List[str],
        groups1: List[str],
        groups2: List[str],
    ) -> Dict[str, float]:
        combined_refs = defaultdict(list)
        combined_hyps = defaultdict(list)
        for r, h, g1, g2 in zip(refs, hyps, groups1, groups2):
            key = f"{g1}_{g2}"
            combined_refs[key].append(r)
            combined_hyps[key].append(h)
        return {
            k: compute_wer(combined_refs[k], combined_hyps[k])
            for k in combined_refs
            if len(combined_refs[k]) >= 5  # min samples for reliability
        }

    def _intersectional_wer_3way(
        self,
        refs, hyps, accents, genders, ages
    ) -> Dict[str, float]:
        combined_refs = defaultdict(list)
        combined_hyps = defaultdict(list)
        for r, h, acc, gen, age in zip(refs, hyps, accents, genders, ages):
            key = f"{acc}_{gen}_{age}"
            combined_refs[key].append(r)
            combined_hyps[key].append(h)
        return {
            k: compute_wer(combined_refs[k], combined_hyps[k])
            for k in combined_refs
            if len(combined_refs[k]) >= 3
        }

    def plot_per_accent_wer(
        self,
        results_dict: Dict[str, Dict],
        save_path: str = None,
    ):
        """
        Bar chart comparing per-accent WER across multiple models.

        Args:
            results_dict: {model_name: fairness_results}
        """
        all_accents = set()
        for r in results_dict.values():
            all_accents.update(r["per_accent_wer"].keys())
        all_accents = sorted(all_accents)

        x = np.arange(len(all_accents))
        width = 0.8 / len(results_dict)
        fig, ax = plt.subplots(figsize=(12, 6))

        for i, (model_name, results) in enumerate(results_dict.items()):
            wers = [results["per_accent_wer"].get(acc, 0) * 100 for acc in all_accents]
            bars = ax.bar(x + i * width, wers, width, label=model_name, alpha=0.8)

        ax.set_xlabel("Accent Group", fontsize=12)
        ax.set_ylabel("Word Error Rate (%)", fontsize=12)
        ax.set_title("Per-Accent WER Comparison", fontsize=14)
        ax.set_xticks(x + width * (len(results_dict) - 1) / 2)
        ax.set_xticklabels(all_accents, rotation=30, ha="right")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()

        save_path = save_path or f"{self.output_dir}/per_accent_wer.png"
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: {save_path}")

    def plot_intersectional_heatmap(
        self,
        intersectional_wer: Dict[str, float],
        title: str = "Intersectional WER (Accent × Gender)",
        save_path: str = None,
    ):
        """Heatmap of WER across intersectional groups."""
        # Parse keys: "accent_gender"
        data = {}
        for key, wer in intersectional_wer.items():
            parts = key.rsplit("_", 1)
            if len(parts) == 2:
                accent, group2 = parts
                if accent not in data:
                    data[accent] = {}
                data[accent][group2] = wer * 100

        df = pd.DataFrame(data).T.fillna(0)
        if df.empty:
            return

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(
            df, annot=True, fmt=".1f", cmap="RdYlGn_r",
            ax=ax, cbar_kws={"label": "WER (%)"}
        )
        ax.set_title(title, fontsize=13)
        plt.tight_layout()

        save_path = save_path or f"{self.output_dir}/intersectional_heatmap.png"
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: {save_path}")

    def save_report(self, all_results: Dict, filename: str = "fairness_report.json"):
        path = f"{self.output_dir}/{filename}"
        with open(path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"Fairness report saved to {path}")
