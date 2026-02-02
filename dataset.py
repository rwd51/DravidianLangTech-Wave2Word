"""
Dataset classes for Tamil Dialect Classification
"""
import random
import numpy as np
import librosa
from torch.utils.data import Dataset

from config import (
    SAMPLING_RATE,
    TIME_STRETCH_PROB,
    PITCH_SHIFT_PROB,
    NOISE_PROB,
    VOLUME_SHIFT_PROB,
    RANDOM_CROP_PROB
)


class AudioAugmenter:
    """
    Lightweight audio augmenter using only librosa + numpy
    """

    def __init__(self, sampling_rate: int = 16000):
        self.sampling_rate = sampling_rate

    def time_stretch(self, audio: np.ndarray, rate: float) -> np.ndarray:
        """Time stretch the audio."""
        return librosa.effects.time_stretch(audio, rate=rate)

    def pitch_shift(self, audio: np.ndarray, n_steps: float) -> np.ndarray:
        """Shift the pitch of the audio."""
        return librosa.effects.pitch_shift(
            audio, sr=self.sampling_rate, n_steps=n_steps
        )

    def add_noise(self, audio: np.ndarray, noise_level: float = 0.005) -> np.ndarray:
        """Add Gaussian noise to the audio."""
        noise = np.random.normal(0, noise_level, audio.shape)
        return audio + noise

    def volume_shift(self, audio: np.ndarray, db_change: float) -> np.ndarray:
        """Change the volume by db_change decibels."""
        factor = 10 ** (db_change / 20)
        return audio * factor

    def random_crop(self, audio: np.ndarray, max_crop_ratio: float = 0.1) -> np.ndarray:
        """Randomly crop the audio from the beginning."""
        if len(audio) < self.sampling_rate:  # Skip if less than 1 second
            return audio
        crop_samples = int(len(audio) * random.uniform(0, max_crop_ratio))
        return audio[crop_samples:]

    def apply_random_augmentations(self, audio: np.ndarray) -> np.ndarray:
        """Apply random augmentations to the audio."""
        augmented = audio.copy()

        # Time stretching (0.9x to 1.1x speed)
        if random.random() < TIME_STRETCH_PROB:
            rate = random.uniform(0.9, 1.1)
            try:
                augmented = self.time_stretch(augmented, rate)
            except Exception:
                pass  # Keep original if stretching fails

        # Pitch shifting (+/- 2 semitones)
        if random.random() < PITCH_SHIFT_PROB:
            n_steps = random.uniform(-2, 2)
            try:
                augmented = self.pitch_shift(augmented, n_steps)
            except Exception:
                pass

        # Add slight noise
        if random.random() < NOISE_PROB:
            noise_level = random.uniform(0.001, 0.01)
            augmented = self.add_noise(augmented, noise_level)

        # Volume change (+/- 6dB)
        if random.random() < VOLUME_SHIFT_PROB:
            db_change = random.uniform(-6, 6)
            augmented = self.volume_shift(augmented, db_change)

        # Random crop (up to 10%)
        if random.random() < RANDOM_CROP_PROB and len(augmented) > self.sampling_rate:
            try:
                augmented = self.random_crop(augmented, max_crop_ratio=0.1)
            except Exception:
                pass

        return augmented


class TamilDialectDataset(Dataset):
    """
    Dataset class for Tamil Dialect Classification + ASR
    Combines regional classification with speech recognition
    """

    def __init__(
        self,
        audio_files: list,
        transcriptions: list,
        dialects: list,
        processor,
        dialect_to_idx: dict,
        sampling_rate: int = SAMPLING_RATE,
        augment: bool = False
    ):
        """
        Initialize the Tamil Dialect Dataset.

        Args:
            audio_files: List of paths to audio files
            transcriptions: List of transcription texts
            dialects: List of dialect labels (e.g., "Northern_Dialect")
            processor: Whisper processor for feature extraction
            dialect_to_idx: Mapping from dialect name to index
            sampling_rate: Audio sampling rate
            augment: Whether to apply augmentation
        """
        self.audio_files = audio_files
        self.transcriptions = transcriptions
        self.dialects = dialects
        self.processor = processor
        self.dialect_to_idx = dialect_to_idx
        self.sampling_rate = sampling_rate
        self.augment = augment

        # Create augmenter if needed
        if self.augment:
            self.augmenter = AudioAugmenter(sampling_rate)
        else:
            self.augmenter = None

        # Validation
        assert len(audio_files) == len(transcriptions) == len(dialects), \
            "Mismatched lengths of audio_files, transcriptions, and dialects"

    def __len__(self) -> int:
        return len(self.audio_files)

    def __getitem__(self, idx: int) -> dict:
        # Load audio file
        audio_path = self.audio_files[idx]
        waveform, sr = librosa.load(audio_path, sr=self.sampling_rate)

        transcription = self.transcriptions[idx]
        dialect = self.dialects[idx]
        dialect_idx = self.dialect_to_idx[dialect]

        # Apply augmentation only during training
        if self.augment and self.augmenter:
            waveform = self.augmenter.apply_random_augmentations(waveform)

        # Preprocess audio using feature extractor
        input_features = self.processor.feature_extractor(
            waveform,
            sampling_rate=self.sampling_rate
        ).input_features[0]

        # Tokenize labels (transcription)
        labels = self.processor.tokenizer(transcription).input_ids

        return {
            "input_features": input_features,
            "labels": labels,
            "region_labels": dialect_idx,  # For classification loss
            "region_idx": dialect_idx,     # For regional adaptation
            "audio_path": audio_path
        }

    @property
    def num_dialects(self) -> int:
        """Return the number of unique dialects."""
        return len(self.dialect_to_idx)

    @property
    def unique_dialects(self) -> list:
        """Return list of unique dialect names."""
        return list(self.dialect_to_idx.keys())


class TamilDialectTestDataset(Dataset):
    """
    Dataset class for Tamil Dialect Classification inference (no labels)
    """

    def __init__(
        self,
        audio_files: list,
        processor,
        sampling_rate: int = SAMPLING_RATE
    ):
        """
        Initialize the Tamil Dialect Test Dataset.

        Args:
            audio_files: List of paths to audio files
            processor: Whisper processor for feature extraction
            sampling_rate: Audio sampling rate
        """
        self.audio_files = audio_files
        self.processor = processor
        self.sampling_rate = sampling_rate

    def __len__(self) -> int:
        return len(self.audio_files)

    def __getitem__(self, idx: int) -> dict:
        # Load audio file
        audio_path = self.audio_files[idx]
        waveform, sr = librosa.load(audio_path, sr=self.sampling_rate)

        # Preprocess audio using feature extractor
        input_features = self.processor.feature_extractor(
            waveform,
            sampling_rate=self.sampling_rate
        ).input_features[0]

        return {
            "input_features": input_features,
            "audio_path": audio_path
        }
