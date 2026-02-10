#!/usr/bin/env python3
"""
Inference script for running transcription on test audio files.
Uses fine-tuned Dolphin model to generate transcriptions.
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
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

# ================== CONFIGURATION ==================
MODEL_DIR = Path("/kaggle/dolphin-finetuned-output")
TEST_AUDIO_DIR = Path("/kaggle/input/datasets/shadabtanjeed/acl2026-tamildialectclassification-og-2/Test")
OUTPUT_FILE = Path("/kaggle/output/transcriptions.txt")
SAMPLE_RATE = 16000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = "float32"
# ===================================================

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
        print(f"Loading model from {self.model_dir}")

        config_path = self.model_dir / "config.yaml"
        feats_stats_path = self.model_dir / "feats_stats.npz"
        bpe_model_path = self.model_dir / "bpe.model"
        model_file = self.model_dir / "model.pt"

        assert config_path.exists(), f"Config not found: {config_path}"
        assert feats_stats_path.exists(), f"Feats stats not found: {feats_stats_path}"
        assert bpe_model_path.exists(), f"BPE model not found: {bpe_model_path}"

        # Ensure feats_stats.npz exists in the expected location
        # Create symlink to avoid issues with hardcoded paths in the config
        expected_stats_path = Path("/kaggle/tamil-game/dolphin/dolphin/assets/feats_stats.npz")
        expected_stats_path.parent.mkdir(parents=True, exist_ok=True)
        
        if not expected_stats_path.exists():
            print(f"Creating symlink: {feats_stats_path} -> {expected_stats_path}")
            try:
                expected_stats_path.symlink_to(feats_stats_path)
            except FileExistsError:
                print(f"Symlink already exists at {expected_stats_path}")

        # Also ensure bpe.model symlink exists
        expected_bpe_path = Path("/kaggle/tamil-game/dolphin/dolphin/assets/bpe.model")
        if not expected_bpe_path.exists():
            print(f"Creating symlink: {bpe_model_path} -> {expected_bpe_path}")
            try:
                expected_bpe_path.symlink_to(bpe_model_path)
            except FileExistsError:
                print(f"Symlink already exists at {expected_bpe_path}")

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

        print("Model loaded successfully")

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
            results = self.model(audio_data)
            
            # Debug: print raw results
            print(f"[DEBUG] Raw results type: {type(results)}")
            print(f"[DEBUG] Raw results: {results}")

            # Extract text and remove special tokens
            text = results.text if hasattr(results, "text") else str(results)
            print(f"[DEBUG] Extracted text: '{text}'")

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
            print(f"[DEBUG] Final text: '{text}'")

            return text
        except Exception as e:
            print(f"[ERROR] Error during transcription: {e}")
            import traceback
            traceback.print_exc()
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
    """Write transcriptions to file in format: filename transcription (no headers)."""
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    print(f"Writing {len(results)} transcriptions to {output_file}")

    with open(output_file, "w", encoding="utf-8") as f:
        for result in results:
            filename = result["filename"]  # Without extension
            predicted = result["predicted"]

            # Write: filename<space>transcription
            f.write(f"{filename} {predicted}\n")

    print(f"Transcriptions written to {output_file}")


def load_test_audio_files(test_dir: Path) -> List[Tuple[str, np.ndarray]]:
    """
    Load all wav files from test directory.
    
    Returns:
        List of tuples: (filename_without_extension, audio_data)
    """
    print(f"Loading audio files from {test_dir}")
    
    audio_files = sorted(test_dir.glob("*.wav"))
    print(f"Found {len(audio_files)} audio files")
    
    samples = []
    for wav_file in audio_files:
        try:
            audio_data = load_audio(str(wav_file), sr=SAMPLE_RATE)
            filename = wav_file.stem  # Filename without extension
            samples.append((filename, audio_data))
        except Exception as e:
            print(f"Warning: Error loading {wav_file}: {e}")
            continue
    
    print(f"Successfully loaded {len(samples)} audio files")
    return samples


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
    print("=" * 80)
    print("DOLPHIN INFERENCE - TEST TRANSCRIPTION")
    print("=" * 80)
    
    # Verify paths exist
    if not MODEL_DIR.exists():
        print(f"ERROR: Model directory not found: {MODEL_DIR}")
        sys.exit(1)
    
    if not TEST_AUDIO_DIR.exists():
        print(f"ERROR: Test audio directory not found: {TEST_AUDIO_DIR}")
        sys.exit(1)
    
    print(f"Model directory: {MODEL_DIR}")
    print(f"Test audio directory: {TEST_AUDIO_DIR}")
    print(f"Output file: {OUTPUT_FILE}")
    
    # Initialize inference engine
    inference_engine = InferenceEngine(MODEL_DIR, device=DEVICE, beam_size=5)
    
    # Load test audio files
    test_samples = load_test_audio_files(TEST_AUDIO_DIR)
    
    if not test_samples:
        print("ERROR: No audio files loaded from test directory")
        sys.exit(1)
    
    print(f"Loaded {len(test_samples)} audio files")
    
    # Transcribe all test files
    print("\nStarting transcription...")
    results = []
    
    for idx, (filename, audio_data) in enumerate(tqdm(test_samples, desc="Transcribing")):
        print(f"\n--- Processing file {idx+1}/{len(test_samples)}: {filename} ---")
        predicted = inference_engine.transcribe(audio_data, sr=SAMPLE_RATE)
        
        print(f"Result for {filename}: '{predicted}'")
        
        results.append({
            "filename": filename,
            "predicted": predicted,
        })
    
    # Write results to file
    print("\nWriting results to file...")
    write_transcriptions(results, str(OUTPUT_FILE))
    
    print("=" * 80)
    print("Inference completed successfully!")
    print(f"Results saved to: {OUTPUT_FILE}")
    print("=" * 80)

