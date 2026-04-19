"""
Data preparation script: download datasets and create JSONL manifests.
Supports: LibriSpeech (clean-100), CommonVoice (English accents), L2-ARCTIC.

Usage:
    python scripts/prepare_data.py --dataset librispeech --data_dir data/
    python scripts/prepare_data.py --dataset commonvoice --data_dir data/ --cv_dir /path/to/cv-corpus
    python scripts/prepare_data.py --dataset l2arctic --data_dir data/ --l2arctic_dir /path/to/l2arctic
"""

import os
import sys
import json
import argparse
import csv
import random
from pathlib import Path
from typing import List, Dict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils.helpers import get_logger, set_seed


# Accent mapping for CommonVoice accents field
CV_ACCENT_MAP = {
    "United States English": "american",
    "England English": "british",
    "Indian English": "indian",
    "Chinese (China)": "chinese",
    "Arabic": "arabic",
    "Korean": "korean",
    "Russian": "russian",
    "Spain Spanish": "spanish",
}

# L2-ARCTIC speakers -> native language -> accent
L2ARCTIC_SPEAKER_MAP = {
    "ABA": "arabic", "SKA": "arabic", "YBAA": "arabic", "ZHAA": "arabic",
    "BWC": "chinese", "LXC": "chinese", "NCC": "chinese", "TXHC": "chinese",
    "ASI": "indian", "RRBI": "indian", "SVBI": "indian", "TNI": "indian",
    "HJK": "korean", "HKK": "korean", "YDCK": "korean", "YKWK": "korean",
    "EBVS": "spanish", "ERMS": "spanish", "MBMPS": "spanish", "NJS": "spanish",
    "HQTV": "vietnamese", "PNV": "vietnamese", "THV": "vietnamese", "TLV": "vietnamese",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Dataset preparation")
    parser.add_argument("--dataset", choices=["librispeech", "commonvoice", "l2arctic", "all"],
                        default="all")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--cv_dir", type=str, default=None, help="CommonVoice corpus root")
    parser.add_argument("--l2arctic_dir", type=str, default=None, help="L2-ARCTIC root")
    parser.add_argument("--librispeech_dir", type=str, default=None, help="LibriSpeech root")
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--test_split", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def write_manifest(samples: List[Dict], path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"  Wrote {len(samples)} samples -> {path}")


def prepare_librispeech(librispeech_dir: str, out_dir: str, logger):
    """Parse LibriSpeech directory structure into JSONL manifests."""
    logger.info("Preparing LibriSpeech...")
    samples = []
    ls_dir = Path(librispeech_dir)

    for trans_file in ls_dir.rglob("*.trans.txt"):
        speaker_dir = trans_file.parent
        with open(trans_file) as f:
            for line in f:
                parts = line.strip().split(" ", 1)
                if len(parts) != 2:
                    continue
                utt_id, text = parts
                audio_path = str(speaker_dir / f"{utt_id}.flac")
                if os.path.exists(audio_path):
                    samples.append({
                        "audio_path": audio_path,
                        "text": text.strip(),
                        "accent": "american",
                        "gender": "unknown",
                        "age": "unknown",
                        "dataset": "librispeech",
                    })

    random.shuffle(samples)
    n_val = int(len(samples) * 0.05)
    n_test = int(len(samples) * 0.05)

    write_manifest(samples[n_val + n_test:], f"{out_dir}/manifests/librispeech_train.jsonl")
    write_manifest(samples[:n_val], f"{out_dir}/manifests/librispeech_val.jsonl")
    write_manifest(samples[n_val:n_val + n_test], f"{out_dir}/manifests/librispeech_test.jsonl")
    logger.info(f"  LibriSpeech: {len(samples)} total samples")
    return samples


def prepare_commonvoice(cv_dir: str, out_dir: str, logger):
    """Parse CommonVoice TSV files into JSONL manifests."""
    logger.info("Preparing CommonVoice...")
    samples = []
    cv_dir = Path(cv_dir)
    clips_dir = cv_dir / "clips"

    for split in ["train", "dev", "test"]:
        tsv_path = cv_dir / f"{split}.tsv"
        if not tsv_path.exists():
            continue

        with open(tsv_path, encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                accent_raw = row.get("accents", "")
                accent = CV_ACCENT_MAP.get(accent_raw, "unknown")
                if accent == "unknown":
                    continue

                audio_path = str(clips_dir / row["path"])
                if not os.path.exists(audio_path):
                    # Try replacing extension
                    audio_path = audio_path.replace(".mp3", ".wav")
                    if not os.path.exists(audio_path):
                        continue

                samples.append({
                    "audio_path": audio_path,
                    "text": row.get("sentence", "").strip(),
                    "accent": accent,
                    "gender": row.get("gender", "unknown"),
                    "age": row.get("age", "unknown"),
                    "dataset": "commonvoice",
                })

    random.shuffle(samples)
    n_val = int(len(samples) * 0.1)
    n_test = int(len(samples) * 0.1)

    write_manifest(samples[n_val + n_test:], f"{out_dir}/manifests/cv_train.jsonl")
    write_manifest(samples[:n_val], f"{out_dir}/manifests/cv_val.jsonl")
    write_manifest(samples[n_val:n_val + n_test], f"{out_dir}/manifests/cv_test.jsonl")
    logger.info(f"  CommonVoice: {len(samples)} accented samples")
    return samples


def prepare_l2arctic(l2arctic_dir: str, out_dir: str, logger):
    """Parse L2-ARCTIC directory into JSONL manifests."""
    logger.info("Preparing L2-ARCTIC...")
    samples = []
    l2_dir = Path(l2arctic_dir)

    for speaker_dir in l2_dir.iterdir():
        speaker_id = speaker_dir.name
        accent = L2ARCTIC_SPEAKER_MAP.get(speaker_id, "unknown")
        if accent == "unknown":
            continue

        transcript_dir = speaker_dir / "transcript"
        wav_dir = speaker_dir / "wav"

        if not transcript_dir.exists() or not wav_dir.exists():
            continue

        for trans_file in transcript_dir.glob("*.txt"):
            utt_id = trans_file.stem
            wav_file = wav_dir / f"{utt_id}.wav"
            if not wav_file.exists():
                continue

            with open(trans_file) as f:
                text = f.read().strip()

            samples.append({
                "audio_path": str(wav_file),
                "text": text,
                "accent": accent,
                "gender": "unknown",
                "age": "unknown",
                "speaker_id": speaker_id,
                "dataset": "l2arctic",
            })

    random.shuffle(samples)
    n_val = int(len(samples) * 0.1)
    n_test = int(len(samples) * 0.15)  # Larger test set for accent evaluation

    write_manifest(samples[n_val + n_test:], f"{out_dir}/manifests/l2arctic_train.jsonl")
    write_manifest(samples[:n_val], f"{out_dir}/manifests/l2arctic_val.jsonl")
    write_manifest(samples[n_val:n_val + n_test], f"{out_dir}/manifests/l2arctic_test.jsonl")
    logger.info(f"  L2-ARCTIC: {len(samples)} samples across {len(L2ARCTIC_SPEAKER_MAP)} speakers")
    return samples


def merge_manifests(manifest_files: List[str], output_path: str):
    """Merge multiple manifest files."""
    all_samples = []
    for path in manifest_files:
        if os.path.exists(path):
            with open(path) as f:
                for line in f:
                    all_samples.append(json.loads(line))
    random.shuffle(all_samples)
    write_manifest(all_samples, output_path)


def main():
    args = parse_args()
    set_seed(args.seed)
    logger = get_logger("prepare_data")
    os.makedirs(args.data_dir, exist_ok=True)

    if args.dataset in ("librispeech", "all") and args.librispeech_dir:
        prepare_librispeech(args.librispeech_dir, args.data_dir, logger)

    if args.dataset in ("commonvoice", "all") and args.cv_dir:
        prepare_commonvoice(args.cv_dir, args.data_dir, logger)

    if args.dataset in ("l2arctic", "all") and args.l2arctic_dir:
        prepare_l2arctic(args.l2arctic_dir, args.data_dir, logger)

    # Merge accented datasets into combined manifests
    accent_train_files = [
        f"{args.data_dir}/manifests/cv_train.jsonl",
        f"{args.data_dir}/manifests/l2arctic_train.jsonl",
    ]
    accent_val_files = [
        f"{args.data_dir}/manifests/cv_val.jsonl",
        f"{args.data_dir}/manifests/l2arctic_val.jsonl",
    ]
    accent_test_files = [
        f"{args.data_dir}/manifests/cv_test.jsonl",
        f"{args.data_dir}/manifests/l2arctic_test.jsonl",
    ]

    merge_manifests(accent_train_files, f"{args.data_dir}/manifests/accented_train.jsonl")
    merge_manifests(accent_val_files, f"{args.data_dir}/manifests/accented_val.jsonl")
    merge_manifests(accent_test_files, f"{args.data_dir}/manifests/accented_test.jsonl")

    logger.info("Data preparation complete.")
    logger.info(f"Manifests saved in: {args.data_dir}/manifests/")


if __name__ == "__main__":
    main()
