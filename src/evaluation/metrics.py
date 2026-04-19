"""
Evaluation metrics for accent-robust ASR.
Primary metric: Word Error Rate (WER) and per-accent ΔWERmax.
"""

import re
import numpy as np
from typing import List, Dict, Optional, Tuple
from collections import defaultdict

try:
    import jiwer
    HAS_JIWER = True
except ImportError:
    HAS_JIWER = False


def normalize_text(text: str) -> str:
    """Lowercase, remove punctuation except apostrophes, collapse whitespace."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9' ]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def compute_wer(references: List[str], hypotheses: List[str]) -> float:
    """
    Compute overall Word Error Rate.

    WER = (S + D + I) / N
    S = substitutions, D = deletions, I = insertions, N = reference words
    """
    refs = [normalize_text(r) for r in references]
    hyps = [normalize_text(h) for h in hypotheses]

    if HAS_JIWER:
        transform = jiwer.Compose([
            jiwer.ToLowerCase(),
            jiwer.RemovePunctuation(),
            jiwer.RemoveMultipleSpaces(),
            jiwer.Strip(),
            jiwer.SentencesToListOfWords(),
        ])
        return jiwer.wer(refs, hyps,
                         reference_transform=transform,
                         hypothesis_transform=transform)
    else:
        return _compute_wer_manual(refs, hyps)


def _compute_wer_manual(references: List[str], hypotheses: List[str]) -> float:
    """Manual WER computation using edit distance."""
    total_errors = 0
    total_words = 0
    for ref, hyp in zip(references, hypotheses):
        ref_words = ref.split()
        hyp_words = hyp.split()
        total_errors += _edit_distance(ref_words, hyp_words)
        total_words += len(ref_words)
    return total_errors / max(total_words, 1)


def _edit_distance(ref: List[str], hyp: List[str]) -> int:
    """Levenshtein edit distance between two word sequences."""
    m, n = len(ref), len(hyp)
    dp = np.zeros((m + 1, n + 1), dtype=int)
    dp[:, 0] = np.arange(m + 1)
    dp[0, :] = np.arange(n + 1)
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref[i - 1] == hyp[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
    return dp[m][n]


def compute_per_accent_wer(
    references: List[str],
    hypotheses: List[str],
    accents: List[str],
) -> Dict[str, float]:
    """
    Compute WER separately for each accent group.

    Returns:
        Dict mapping accent name to WER float.
    """
    accent_refs = defaultdict(list)
    accent_hyps = defaultdict(list)

    for ref, hyp, accent in zip(references, hypotheses, accents):
        accent_refs[accent].append(ref)
        accent_hyps[accent].append(hyp)

    per_accent_wer = {}
    for accent in accent_refs:
        per_accent_wer[accent] = compute_wer(accent_refs[accent], accent_hyps[accent])

    return per_accent_wer


def compute_delta_wer_max(per_accent_wer: Dict[str, float]) -> float:
    """
    ΔWERmax = max(WER_i) - min(WER_i)

    Primary fairness metric measuring maximum disparity across accent groups.
    Lower is better (0 = perfectly fair).
    """
    if not per_accent_wer:
        return 0.0
    wers = list(per_accent_wer.values())
    return max(wers) - min(wers)


def compute_character_error_rate(references: List[str], hypotheses: List[str]) -> float:
    """Character Error Rate (CER)."""
    refs = [normalize_text(r).replace(" ", "") for r in references]
    hyps = [normalize_text(h).replace(" ", "") for h in hypotheses]
    total_errors = sum(_edit_distance(list(r), list(h)) for r, h in zip(refs, hyps))
    total_chars = sum(len(r) for r in refs)
    return total_errors / max(total_chars, 1)


def compute_all_metrics(
    references: List[str],
    hypotheses: List[str],
    accents: Optional[List[str]] = None,
    genders: Optional[List[str]] = None,
    ages: Optional[List[str]] = None,
) -> dict:
    """Compute comprehensive evaluation metrics."""
    results = {
        "wer": compute_wer(references, hypotheses),
        "cer": compute_character_error_rate(references, hypotheses),
        "num_samples": len(references),
    }

    if accents:
        per_accent = compute_per_accent_wer(references, hypotheses, accents)
        results["per_accent_wer"] = per_accent
        results["delta_wer_max"] = compute_delta_wer_max(per_accent)

        # Intersectional analysis (accent x gender)
        if genders:
            intersect_refs = defaultdict(list)
            intersect_hyps = defaultdict(list)
            for ref, hyp, acc, gen in zip(references, hypotheses, accents, genders):
                key = f"{acc}_{gen}"
                intersect_refs[key].append(ref)
                intersect_hyps[key].append(hyp)
            results["intersectional_wer"] = {
                k: compute_wer(intersect_refs[k], intersect_hyps[k])
                for k in intersect_refs
            }

    return results


def print_metrics_table(metrics: dict):
    """Pretty-print metrics to stdout."""
    print(f"\n{'='*50}")
    print(f"Overall WER:  {metrics['wer']:.4f} ({metrics['wer']*100:.2f}%)")
    print(f"Overall CER:  {metrics['cer']:.4f} ({metrics['cer']*100:.2f}%)")
    print(f"Num samples:  {metrics['num_samples']}")

    if "per_accent_wer" in metrics:
        print(f"\nPer-Accent WER:")
        for accent, wer in sorted(metrics["per_accent_wer"].items()):
            print(f"  {accent:15s}: {wer:.4f} ({wer*100:.2f}%)")
        print(f"\n  ΔWERmax:     {metrics['delta_wer_max']:.4f} ({metrics['delta_wer_max']*100:.2f}%)")
    print("="*50)
