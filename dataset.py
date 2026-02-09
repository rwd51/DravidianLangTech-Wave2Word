"""
Dataset classes for Tamil Dialect Classification
Enhanced with SpecAugment and curriculum-based augmentation that varies per epoch.

Literature references:
- SpecAugment: Park et al. (2019) "SpecAugment: A Simple Data Augmentation Method for ASR"
- Speed Perturbation: Ko et al. (2015) "Audio Augmentation for Speech Recognition"
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


class SpecAugment:
    """
    SpecAugment: Time and Frequency masking for mel spectrograms.
    From: "SpecAugment: A Simple Data Augmentation Method for ASR" (Park et al., 2019)
    """

    def __init__(
        self,
        freq_mask_param: int = 27,  # F in paper (max freq mask width)
        time_mask_param: int = 100,  # T in paper (max time mask width)
        n_freq_masks: int = 2,  # Number of frequency masks
        n_time_masks: int = 2,  # Number of time masks
        mask_value: float = 0.0  # Value to fill masked regions
    ):
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param
        self.n_freq_masks = n_freq_masks
        self.n_time_masks = n_time_masks
        self.mask_value = mask_value

    def __call__(self, mel_spec: np.ndarray, intensity: float = 1.0) -> np.ndarray:
        """
        Apply SpecAugment to mel spectrogram.

        Args:
            mel_spec: Mel spectrogram of shape (n_mels, time)
            intensity: Augmentation intensity (0.0 to 1.0), scales mask sizes

        Returns:
            Augmented mel spectrogram
        """
        augmented = mel_spec.copy()
        n_mels, n_frames = augmented.shape

        # Scale parameters by intensity
        freq_param = int(self.freq_mask_param * intensity)
        time_param = int(self.time_mask_param * intensity)

        # Frequency masking
        for _ in range(self.n_freq_masks):
            if freq_param > 0:
                f = random.randint(0, min(freq_param, n_mels - 1))
                f0 = random.randint(0, n_mels - f)
                augmented[f0:f0 + f, :] = self.mask_value

        # Time masking
        for _ in range(self.n_time_masks):
            if time_param > 0:
                t = random.randint(0, min(time_param, n_frames - 1))
                t0 = random.randint(0, n_frames - t)
                augmented[:, t0:t0 + t] = self.mask_value

        return augmented


class AudioAugmenter:
    """
    Comprehensive audio augmenter with curriculum learning support.
    Augmentation intensity increases with epoch for curriculum learning.
    """

    def __init__(self, sampling_rate: int = 16000):
        self.sampling_rate = sampling_rate
        self.current_epoch = 0
        self.max_epochs = 20  # For scaling intensity

    def set_epoch(self, epoch: int):
        """Set current epoch for curriculum-based augmentation."""
        self.current_epoch = epoch

    def get_intensity(self) -> float:
        """
        Get augmentation intensity based on current epoch.
        Starts at 0.3 and increases to 1.0 over training.
        Curriculum: start easy, increase difficulty.
        """
        min_intensity = 0.3
        max_intensity = 1.0
        # Linear increase from min to max over epochs
        progress = min(self.current_epoch / max(self.max_epochs - 1, 1), 1.0)
        return min_intensity + (max_intensity - min_intensity) * progress

    def time_stretch(self, audio: np.ndarray, rate: float) -> np.ndarray:
        """Time stretch the audio (speed perturbation)."""
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

    def add_colored_noise(self, audio: np.ndarray, noise_level: float = 0.01) -> np.ndarray:
        """Add pink/brown noise (more realistic environmental noise)."""
        # Generate pink noise using 1/f spectrum
        n_samples = len(audio)
        white = np.random.randn(n_samples)
        # Simple pink noise approximation
        pink = np.cumsum(white) / np.sqrt(n_samples)
        pink = pink / np.std(pink) * noise_level
        return audio + pink

    def volume_shift(self, audio: np.ndarray, db_change: float) -> np.ndarray:
        """Change the volume by db_change decibels."""
        factor = 10 ** (db_change / 20)
        return audio * factor

    def random_crop(self, audio: np.ndarray, max_crop_ratio: float = 0.1) -> np.ndarray:
        """Randomly crop the audio from the beginning."""
        if len(audio) < self.sampling_rate:
            return audio
        crop_samples = int(len(audio) * random.uniform(0, max_crop_ratio))
        return audio[crop_samples:]

    def time_shift(self, audio: np.ndarray, max_shift_ratio: float = 0.1) -> np.ndarray:
        """Randomly shift audio in time (with wraparound)."""
        shift = int(len(audio) * random.uniform(-max_shift_ratio, max_shift_ratio))
        return np.roll(audio, shift)

    def apply_random_augmentations(self, audio: np.ndarray) -> np.ndarray:
        """
        Apply random augmentations to the audio.
        Intensity varies based on current epoch (curriculum learning).
        """
        augmented = audio.copy()
        intensity = self.get_intensity()

        # Scale probabilities by intensity for curriculum effect
        prob_scale = 0.5 + 0.5 * intensity  # Range: 0.65 to 1.0

        # Time stretching (speed perturbation: 0.9x to 1.1x)
        if random.random() < TIME_STRETCH_PROB * prob_scale:
            # Scale stretch range by intensity
            stretch_range = 0.05 + 0.05 * intensity  # 0.05 to 0.10
            rate = random.uniform(1 - stretch_range, 1 + stretch_range)
            try:
                augmented = self.time_stretch(augmented, rate)
            except Exception:
                pass

        # Pitch shifting (+/- semitones scaled by intensity)
        if random.random() < PITCH_SHIFT_PROB * prob_scale:
            max_steps = 1 + 2 * intensity  # 1 to 3 semitones
            n_steps = random.uniform(-max_steps, max_steps)
            try:
                augmented = self.pitch_shift(augmented, n_steps)
            except Exception:
                pass

        # Add noise (Gaussian or colored)
        if random.random() < NOISE_PROB * prob_scale:
            noise_level = random.uniform(0.001, 0.005 + 0.01 * intensity)
            if random.random() < 0.5:
                augmented = self.add_noise(augmented, noise_level)
            else:
                augmented = self.add_colored_noise(augmented, noise_level)

        # Volume change (+/- dB scaled by intensity)
        if random.random() < VOLUME_SHIFT_PROB * prob_scale:
            max_db = 3 + 5 * intensity  # 3 to 8 dB
            db_change = random.uniform(-max_db, max_db)
            augmented = self.volume_shift(augmented, db_change)

        # Time shift (new augmentation)
        if random.random() < 0.3 * prob_scale:
            max_shift = 0.05 + 0.05 * intensity
            augmented = self.time_shift(augmented, max_shift)

        # Random crop (up to 10%)
        if random.random() < RANDOM_CROP_PROB * prob_scale and len(augmented) > self.sampling_rate:
            try:
                max_crop = 0.05 + 0.05 * intensity
                augmented = self.random_crop(augmented, max_crop_ratio=max_crop)
            except Exception:
                pass

        return augmented


class TamilDialectDataset(Dataset):
    """
    Dataset class for Tamil Dialect Classification + ASR
    Combines regional classification with speech recognition.

    Features:
    - Audio-level augmentation (time stretch, pitch shift, noise, etc.)
    - SpecAugment on mel spectrograms (time/frequency masking)
    - Curriculum learning: augmentation intensity increases per epoch
    """

    def __init__(
        self,
        audio_files: list,
        transcriptions: list,
        dialects: list,
        processor,
        dialect_to_idx: dict,
        sampling_rate: int = SAMPLING_RATE,
        augment: bool = False,
        use_specaugment: bool = True,
        max_epochs: int = 20
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
            use_specaugment: Whether to apply SpecAugment on mel spectrograms
            max_epochs: Total epochs for curriculum scaling
        """
        self.audio_files = audio_files
        self.transcriptions = transcriptions
        self.dialects = dialects
        self.processor = processor
        self.dialect_to_idx = dialect_to_idx
        self.sampling_rate = sampling_rate
        self.augment = augment
        self.use_specaugment = use_specaugment and augment
        self.current_epoch = 0
        self.max_epochs = max_epochs

        # Create augmenters if needed
        if self.augment:
            self.audio_augmenter = AudioAugmenter(sampling_rate)
            self.audio_augmenter.max_epochs = max_epochs
            if self.use_specaugment:
                self.spec_augmenter = SpecAugment(
                    freq_mask_param=27,   # Whisper uses 80 mel bins, mask up to 27
                    time_mask_param=100,  # Max 100 frames masked
                    n_freq_masks=2,
                    n_time_masks=2
                )
        else:
            self.audio_augmenter = None
            self.spec_augmenter = None

        # Validation
        assert len(audio_files) == len(transcriptions) == len(dialects), \
            "Mismatched lengths of audio_files, transcriptions, and dialects"

    def set_epoch(self, epoch: int):
        """
        Set current epoch for curriculum-based augmentation.
        Call this at the start of each epoch to adjust augmentation intensity.
        """
        self.current_epoch = epoch
        if self.audio_augmenter:
            self.audio_augmenter.set_epoch(epoch)

    def get_augmentation_intensity(self) -> float:
        """Get current augmentation intensity based on epoch."""
        if self.audio_augmenter:
            return self.audio_augmenter.get_intensity()
        return 0.0

    def __len__(self) -> int:
        return len(self.audio_files)

    def __getitem__(self, idx: int) -> dict:
        # Load audio file
        audio_path = self.audio_files[idx]
        waveform, sr = librosa.load(audio_path, sr=self.sampling_rate)

        transcription = self.transcriptions[idx]
        dialect = self.dialects[idx]
        dialect_idx = self.dialect_to_idx[dialect]

        # Apply audio-level augmentation (curriculum-based)
        if self.augment and self.audio_augmenter:
            waveform = self.audio_augmenter.apply_random_augmentations(waveform)

        # Preprocess audio using feature extractor
        input_features = self.processor.feature_extractor(
            waveform,
            sampling_rate=self.sampling_rate
        ).input_features[0]

        # Apply SpecAugment on mel spectrogram (curriculum-based)
        if self.use_specaugment and self.spec_augmenter:
            intensity = self.get_augmentation_intensity()
            # Only apply with probability that increases with epoch
            if random.random() < 0.5 + 0.3 * intensity:
                input_features = self.spec_augmenter(input_features, intensity=intensity)

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
