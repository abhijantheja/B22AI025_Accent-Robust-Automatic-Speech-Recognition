"""
Grad-CAM explainability visualization for accent-robust ASR.
Highlights temporal regions of speech influencing model decisions.

Usage:
    python scripts/visualize_gradcam.py \
        --checkpoint results/phase2_adversarial/best_model.pt \
        --audio_path data/samples/indian_speaker.wav \
        --accent indian \
        --output_dir results/gradcam
"""

import os
import sys
import argparse
import torch
import numpy as np
import librosa

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.adversarial_asr import AdversarialASRModel
from src.data.preprocessing import AudioPreprocessor
from src.evaluation.explainability import GradCAMVisualizer
from src.evaluation.metrics import compute_wer
from src.utils.helpers import set_seed, get_logger, load_checkpoint
import yaml


def parse_args():
    parser = argparse.ArgumentParser(description="Grad-CAM Visualization")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--audio_path", type=str, required=True)
    parser.add_argument("--reference_text", type=str, default="")
    parser.add_argument("--accent", type=str, default="unknown")
    parser.add_argument("--config", type=str, default="configs/adversarial_config.yaml")
    parser.add_argument("--output_dir", type=str, default="results/gradcam")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    logger = get_logger("visualize_gradcam")
    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.config, "r") as f:
        import yaml
        config = yaml.safe_load(f)

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    preprocessor = AudioPreprocessor(processor_name=config.get("model_name", "facebook/wav2vec2-base"))

    model = AdversarialASRModel(
        model_name=config.get("model_name", "facebook/wav2vec2-base"),
        vocab_size=config.get("vocab_size", 32),
        num_accents=config.get("num_accents", 8),
    )
    model, _, _ = load_checkpoint(args.checkpoint, model, device=device)
    model = model.to(device)
    logger.info(f"Model loaded from {args.checkpoint}")

    # Load audio
    waveform, sr = preprocessor.load_audio(args.audio_path)
    input_values = preprocessor.process(waveform, sr).unsqueeze(0).to(device)

    # Get transcription
    with torch.no_grad():
        ctc_logits, sup_logits, _ = model(input_values=input_values, use_adversarial=False)

    pred_ids = ctc_logits.argmax(dim=-1)
    transcription = preprocessor.processor.batch_decode(pred_ids)[0]
    predicted_accent_id = sup_logits.argmax(dim=-1).item()

    logger.info(f"Predicted transcription: '{transcription}'")
    logger.info(f"Reference: '{args.reference_text}'")
    if args.reference_text:
        wer = compute_wer([args.reference_text], [transcription])
        logger.info(f"WER: {wer:.4f}")

    # Grad-CAM
    visualizer = GradCAMVisualizer(model, target_layer_name="wav2vec2.encoder.layers")
    cam = visualizer.compute_gradcam(input_values, target_class=predicted_accent_id)

    fname = os.path.splitext(os.path.basename(args.audio_path))[0]
    save_path = os.path.join(args.output_dir, f"gradcam_{fname}_{args.accent}.png")

    visualizer.visualize(
        waveform=waveform,
        cam=cam,
        sr=sr,
        transcription=transcription,
        accent=args.accent,
        save_path=save_path,
        show=False,
    )

    visualizer.remove_hooks()
    logger.info(f"Grad-CAM saved: {save_path}")


if __name__ == "__main__":
    main()
