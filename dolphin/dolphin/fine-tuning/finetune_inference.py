#!/usr/bin/env python3
"""
Utility for running inference on trained models and generating transcriptions.
"""

import os
import sys
import logging
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from finetune_config import (
    SAMPLE_RATE,
    DEVICE,
    DTYPE,
    format_token_sequence,
    TRANSCRIPTION_OUTPUT,
)
from finetune_dataclass import AudioSample
from dolphin.model import DolphinSpeech2Text
from dolphin.audio import load_audio

logger = logging.getLogger(__name__)

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s"
)


class InferenceEngine:
    """Inference engine for transcription."""

    def __init__(self, model_dir: Path, device: str = DEVICE, beam_size: int = 5):
        """Initialize inference engine with fine-tuned model."""
        self.model_dir = Path(model_dir)
        self.device = device
        self.beam_size = beam_size
        self.model = None
        self.load_model()

    def load_model(self):
        """Load fine-tuned model."""
        logger.info(f"Loading model from {self.model_dir}")

        config_path = self.model_dir / "config.yaml"
        bpe_model_path = self.model_dir / "bpe.model"
        feats_stats_path = self.model_dir / "feats_stats.npz"
        model_file = self.model_dir / "small.pt"

        assert config_path.exists(), f"Config not found: {config_path}"
        assert bpe_model_path.exists(), f"BPE model not found: {bpe_model_path}"
        assert feats_stats_path.exists(), f"Feats stats not found: {feats_stats_path}"

        # Load model using DolphinSpeech2Text
        self.model = DolphinSpeech2Text(
            s2t_train_config=str(config_path),
            s2t_model_file=str(model_file) if model_file.exists() else None,
            device=self.device,
            dtype=DTYPE,
            beam_size=self.beam_size,
            nbest=1,
            task_sym="<asr>",
            predict_time=True,
        )

        logger.info("Model loaded successfully")

    def transcribe(self, audio_data: np.ndarray, sr: int = SAMPLE_RATE) -> str:
        """
        Transcribe audio data.

        Args:
            audio_data: Audio waveform as numpy array
            sr: Sample rate

        Returns:
            Transcription text without special tokens
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")

        try:
            # Use model's transcribe method
            results = self.model(audio_data, sr=sr)

            # Extract text and remove special tokens
            text = results.text if hasattr(results, "text") else str(results)

            # Remove special tokens
            for token in [
                "<SOS>",
                "<LANGUAGE>",
                "<REGION>",
                "<TRANSCRIBE>",
                "<EOS>",
                "<asr>",
            ]:
                text = text.replace(token, "").replace(f"<", "").replace(f">", "")

            # Clean up extra whitespace
            text = " ".join(text.split())

            return text
        except Exception as e:
            logger.warning(f"Error during transcription: {e}")
            return ""

    def transcribe_samples(self, samples: List[AudioSample]) -> List[Dict]:
        """
        Transcribe multiple audio samples.

        Returns:
            List of dicts with keys: filenames, original, predicted
        """
        logger.info(f"Transcribing {len(samples)} samples...")

        results = []

        for sample in tqdm(samples, desc="Transcribing"):
            predicted = self.transcribe(sample.audio_data)

            results.append(
                {
                    "filenames": sample.original_filenames,
                    "original": sample.transcription,
                    "predicted": predicted,
                    "duration": sample.duration,
                    "dialect": sample.dialect,
                }
            )

        return results


def write_transcriptions(results: List[Dict], output_file: str):
    """Write transcriptions to file."""
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Writing {len(results)} transcriptions to {output_file}")

    with open(output_file, "w", encoding="utf-8") as f:
        f.write("filename\toriginal_transcription\tmodel_transcription\n")
        f.write("-" * 150 + "\n")

        for result in results:
            filenames = ", ".join(result["filenames"])
            original = result["original"]
            predicted = result["predicted"]

            # Escape any special characters for file output
            original = original.replace("\t", " ").replace("\n", " ")
            predicted = predicted.replace("\t", " ").replace("\n", " ")

            f.write(f"{filenames}\t{original}\t{predicted}\n")

    logger.info(f"Transcriptions written to {output_file}")


def compute_metrics(results: List[Dict]) -> Dict:
    """Compute WER and other metrics."""
    from finetune_utils import compute_wer

    total_wer = 0.0
    num_samples = 0

    for result in results:
        original = result["original"].lower()
        predicted = result["predicted"].lower()

        if original:
            wer = compute_wer(predicted, original)
            total_wer += wer
            num_samples += 1

    avg_wer = total_wer / num_samples if num_samples > 0 else 0.0

    metrics = {
        "avg_wer": avg_wer,
        "num_samples": num_samples,
    }

    return metrics


if __name__ == "__main__":
    print("Inference utility module. Import and use InferenceEngine class.")
