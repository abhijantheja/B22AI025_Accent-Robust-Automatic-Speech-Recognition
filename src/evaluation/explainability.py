"""
Grad-CAM visualization for accent-robust ASR explainability.
Highlights temporal speech regions most influential for correct/incorrect transcriptions.
Reference: Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks", ICCV 2017.
"""

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import librosa
import librosa.display
from typing import Optional, List, Tuple


class GradCAMVisualizer:
    """
    Grad-CAM for Wav2Vec 2.0 encoder.

    Computes gradient-weighted activation maps over the encoder's
    last hidden states to explain which temporal regions drive predictions.
    """

    def __init__(self, model, target_layer_name: str = "wav2vec2.encoder.layers"):
        self.model = model
        self.target_layer_name = target_layer_name
        self.gradients = None
        self.activations = None
        self._hooks = []
        self._register_hooks()

    def _get_target_layer(self):
        parts = self.target_layer_name.split(".")
        layer = self.model
        for p in parts:
            if p.isdigit():
                layer = layer[int(p)]
            else:
                layer = getattr(layer, p)
        # Use last transformer layer
        return layer[-1]

    def _register_hooks(self):
        target = self._get_target_layer()

        def forward_hook(module, input, output):
            if isinstance(output, tuple):
                self.activations = output[0].detach()
            else:
                self.activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            if isinstance(grad_out, tuple):
                self.gradients = grad_out[0].detach()
            else:
                self.gradients = grad_out.detach()

        self._hooks.append(target.register_forward_hook(forward_hook))
        self._hooks.append(target.register_full_backward_hook(backward_hook))

    def remove_hooks(self):
        for hook in self._hooks:
            hook.remove()
        self._hooks = []

    def compute_gradcam(
        self,
        input_values: torch.Tensor,
        attention_mask: torch.Tensor = None,
        target_class: int = None,
    ) -> np.ndarray:
        """
        Compute Grad-CAM activation map.

        Args:
            input_values: (1, T) waveform tensor
            attention_mask: (1, T) mask
            target_class: Accent class index to explain (None = max predicted class)

        Returns:
            cam: (T',) normalized activation map over encoder time steps
        """
        self.model.eval()
        input_values = input_values.requires_grad_(False)

        # Forward pass
        ctc_logits, sup_logits, _ = self.model(
            input_values=input_values,
            attention_mask=attention_mask,
            use_adversarial=False,
        )

        # Target: accent classifier output
        if target_class is None:
            target_class = sup_logits.argmax(dim=-1).item()

        # Backward on target class score
        self.model.zero_grad()
        score = sup_logits[0, target_class]
        score.backward()

        # Grad-CAM: global average pool gradients over feature dimension
        # activations: (1, T', D), gradients: (1, T', D)
        if self.gradients is None or self.activations is None:
            return np.zeros(ctc_logits.shape[1])

        weights = self.gradients.mean(dim=2, keepdim=True)  # (1, T', 1)
        cam = (weights * self.activations).sum(dim=2)  # (1, T')
        cam = F.relu(cam).squeeze(0).cpu().numpy()  # (T',)

        # Normalize to [0, 1]
        if cam.max() > 0:
            cam = cam / cam.max()

        return cam

    def visualize(
        self,
        waveform: np.ndarray,
        cam: np.ndarray,
        sr: int = 16000,
        transcription: str = "",
        accent: str = "",
        save_path: str = None,
        show: bool = False,
    ):
        """
        Overlay Grad-CAM on mel-spectrogram and waveform.
        """
        n_fft = 400
        hop_length = 160
        n_mels = 80

        mel = librosa.feature.melspectrogram(
            y=waveform, sr=sr, n_mels=n_mels,
            n_fft=n_fft, hop_length=hop_length
        )
        mel_db = librosa.power_to_db(mel, ref=np.max)

        # Interpolate CAM to mel-spectrogram time resolution
        mel_frames = mel_db.shape[1]
        cam_interp = np.interp(
            np.linspace(0, 1, mel_frames),
            np.linspace(0, 1, len(cam)),
            cam
        )

        fig, axes = plt.subplots(3, 1, figsize=(14, 10))

        # 1. Waveform with CAM overlay
        t_wave = np.linspace(0, len(waveform) / sr, len(waveform))
        t_cam = np.linspace(0, len(waveform) / sr, mel_frames)
        axes[0].plot(t_wave, waveform, color="steelblue", alpha=0.7)
        axes[0].fill_between(
            t_cam, -np.abs(waveform).max() * cam_interp,
            np.abs(waveform).max() * cam_interp,
            alpha=0.4, color="red", label="Grad-CAM"
        )
        axes[0].set_title(f"Waveform + Grad-CAM  |  Accent: {accent}", fontsize=12)
        axes[0].set_xlabel("Time (s)")
        axes[0].legend(loc="upper right")

        # 2. Mel-spectrogram
        librosa.display.specshow(
            mel_db, sr=sr, hop_length=hop_length,
            x_axis="time", y_axis="mel",
            ax=axes[1], cmap="magma"
        )
        axes[1].set_title("Mel-Spectrogram", fontsize=12)

        # 3. Grad-CAM heatmap overlay on mel-spectrogram
        librosa.display.specshow(
            mel_db, sr=sr, hop_length=hop_length,
            x_axis="time", y_axis="mel",
            ax=axes[2], cmap="magma", alpha=0.6
        )
        axes[2].imshow(
            cam_interp[np.newaxis, :] * np.ones((n_mels, 1)),
            aspect="auto",
            origin="lower",
            extent=[0, len(waveform) / sr, 0, sr // 2],
            cmap="hot", alpha=0.5,
            vmin=0, vmax=1,
        )
        axes[2].set_title(f"Grad-CAM Overlay  |  '{transcription}'", fontsize=12)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Saved Grad-CAM visualization: {save_path}")
        if show:
            plt.show()
        plt.close()
