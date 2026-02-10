# Data structures and loading for fine-tuning

import os
import logging
import pickle
import hashlib
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import numpy as np
from dataclasses import dataclass
from tqdm import tqdm

from finetune_config import (
    DATASET_ROOT,
    SAMPLE_RATE,
    AUDIO_MIN_DURATION,
    MAX_COMBINED_AUDIO_LENGTH,
    VALIDATION_SPLIT,
    format_token_sequence,
    CACHE_DIR,
    USE_CACHE,
)

# Add parent directory for imports
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dolphin.audio import load_audio

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)


@dataclass
class AudioSample:
    """Represents a single audio sample or combined samples."""

    audio_data: np.ndarray
    transcription: str
    original_filenames: List[str]
    duration: float  # in seconds

    def get_token_sequence(self) -> str:
        """Get formatted token sequence with special tokens."""
        return format_token_sequence(self.transcription)


class AudioDataLoader:
    """Load and preprocess all audio data from dataset root."""

    def __init__(self, dataset_root: str, combine_short_audios: bool = True, use_cache: bool = USE_CACHE):
        """
        Initialize data loader.

        Args:
            dataset_root: Root path to dataset containing audio subfolders and text files
            combine_short_audios: Whether to combine audios shorter than MAX_COMBINED_AUDIO_LENGTH
            use_cache: Whether to use cached processed audio data
        """
        self.dataset_root = Path(dataset_root)
        self.combine_short_audios = combine_short_audios
        self.use_cache = use_cache
        self.samples = []
        self.cache_file = self._get_cache_path()

    def _get_cache_path(self) -> Path:
        """Generate cache file path based on dataset configuration."""
        # Create a hash of the configuration to ensure cache invalidation on config changes
        config_str = f"{self.dataset_root}_{SAMPLE_RATE}_{AUDIO_MIN_DURATION}_{MAX_COMBINED_AUDIO_LENGTH}_{self.combine_short_audios}_{VALIDATION_SPLIT}"
        config_hash = hashlib.md5(config_str.encode()).hexdigest()[:8]
        cache_filename = f"audio_cache_{config_hash}.pkl"
        return Path(CACHE_DIR) / cache_filename
    
    def _save_cache(self, train_samples: List[AudioSample], val_samples: List[AudioSample]):
        """Save processed samples to cache."""
        try:
            logger.info(f"Saving processed audio data to cache: {self.cache_file}")
            cache_data = {
                'train_samples': train_samples,
                'val_samples': val_samples,
                'config': {
                    'dataset_root': str(self.dataset_root),
                    'sample_rate': SAMPLE_RATE,
                    'audio_min_duration': AUDIO_MIN_DURATION,
                    'max_combined_audio_length': MAX_COMBINED_AUDIO_LENGTH,
                    'combine_short_audios': self.combine_short_audios,
                    'validation_split': VALIDATION_SPLIT,
                }
            }
            with open(self.cache_file, 'wb') as f:
                pickle.dump(cache_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            logger.info(f"Cache saved successfully ({self.cache_file.stat().st_size / 1024 / 1024:.2f} MB)")
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")
    
    def _load_cache(self) -> Tuple[Optional[List[AudioSample]], Optional[List[AudioSample]]]:
        """Load processed samples from cache if available."""
        if not self.use_cache:
            return None, None
        
        if not self.cache_file.exists():
            logger.info("No cache file found, will process audio from scratch")
            return None, None
        
        try:
            logger.info(f"Loading cached audio data from: {self.cache_file}")
            with open(self.cache_file, 'rb') as f:
                cache_data = pickle.load(f)
            
            # Verify cache is for the same configuration
            cached_config = cache_data.get('config', {})
            if cached_config.get('dataset_root') != str(self.dataset_root):
                logger.warning("Cache is for different dataset, will reprocess")
                return None, None
            
            train_samples = cache_data['train_samples']
            val_samples = cache_data['val_samples']
            
            logger.info(f"✓ Loaded from cache: {len(train_samples)} train samples, {len(val_samples)} val samples")
            logger.info(f"Cache file size: {self.cache_file.stat().st_size / 1024 / 1024:.2f} MB")
            
            return train_samples, val_samples
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}, will reprocess audio")
            return None, None

    def _get_duration(self, audio_data: np.ndarray) -> float:
        """Calculate audio duration in seconds."""
        return len(audio_data) / SAMPLE_RATE

    def _discover_and_load_samples(self) -> List[Dict]:
        """
        Discover and load all audio files from subfolders in dataset root.

        Looks for folders ending with "_audio" and corresponding "*_Text.txt" files.
        All folders are processed together as a single dataset.

        Returns:
            List of sample dictionaries
        """
        if not self.dataset_root.exists():
            logger.error(f"Dataset root not found: {self.dataset_root}")
            return []

        logger.info(f"Loading audio data from: {self.dataset_root}")

        samples = []

        # Find all audio folders (e.g., sp1_tha_audio, sp2_tha_audio)
        audio_folders = sorted(
            [d for d in self.dataset_root.rglob("*_audio") if d.is_dir()]
        )

        if not audio_folders:
            logger.warning(f"No audio folders found in {self.dataset_root}")
            return []

        logger.info(f"Found {len(audio_folders)} audio folder(s)")

        for audio_folder in audio_folders:
            # Get corresponding text file (e.g., sp1_tha_audio -> sp1_tha_Text.txt)
            base_name = audio_folder.name.replace("_audio", "")
            text_file = audio_folder.parent / f"{base_name}_Text.txt"

            if not text_file.exists():
                logger.warning(f"Text file not found: {text_file}")
                continue

            # Load transcriptions
            transcriptions = {}
            with open(text_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split(None, 1)  # Split on first whitespace
                    if len(parts) == 2:
                        filename, transcription = parts
                        transcriptions[filename] = transcription

            # Load audio files
            wav_files = sorted(audio_folder.glob("*.wav"))

            for wav_file in wav_files:
                filename = wav_file.stem  # Filename without extension
                if filename in transcriptions:
                    try:
                        audio_data = load_audio(str(wav_file), sr=SAMPLE_RATE)
                        duration = self._get_duration(audio_data)

                        # Only include audio above minimum duration
                        if duration >= AUDIO_MIN_DURATION:
                            samples.append(
                                {
                                    "audio": audio_data,
                                    "transcription": transcriptions[filename],
                                    "filenames": [filename],  # Always use list for consistency
                                    "duration": duration,
                                    "path": str(wav_file),
                                }
                            )
                    except Exception as e:
                        logger.warning(f"Error loading {wav_file}: {e}")
                        continue

        logger.info(
            f"Loaded {len(samples)} samples from {len(audio_folders)} audio folder(s)"
        )
        return samples

    def _combine_short_audios(self, samples: List[Dict]) -> List[Dict]:
        """Combine multiple short audio files to form samples of MAX_COMBINED_AUDIO_LENGTH."""
        if not self.combine_short_audios:
            return samples

        logger.info("Combining short audio files...")
        combined_samples = []
        current_batch = []
        current_duration = 0.0

        for sample in tqdm(samples, desc="Combining audios"):
            if current_duration + sample["duration"] <= MAX_COMBINED_AUDIO_LENGTH:
                # Add to current batch
                current_batch.append(sample)
                current_duration += sample["duration"]
            else:
                # Save current batch if not empty
                if current_batch:
                    if len(current_batch) > 1:
                        # Combine multiple audios
                        combined_audio = np.concatenate(
                            [s["audio"] for s in current_batch]
                        )
                        combined_transcription = " ".join(
                            [s["transcription"] for s in current_batch]
                        )
                        combined_filenames = [s["filename"] for s in current_batch]

                        combined_samples.append(
                            {
                                "audio": combined_audio,
                                "transcription": combined_transcription,
                                "filenames": combined_filenames,
                                "duration": current_duration,
                            }
                        )
                    else:
                        # Single audio in batch
                        sample = current_batch[0]
                        combined_samples.append(
                            {
                                "audio": sample["audio"],
                                "transcription": sample["transcription"],
                                "filenames": [sample["filename"]],
                                "duration": sample["duration"],
                            }
                        )

                # Start new batch with current sample
                current_batch = [sample]
                current_duration = sample["duration"]

        # Don't forget the last batch
        if current_batch:
            if len(current_batch) > 1:
                combined_audio = np.concatenate([s["audio"] for s in current_batch])
                combined_transcription = " ".join(
                    [s["transcription"] for s in current_batch]
                )
                combined_filenames = [s["filename"] for s in current_batch]

                combined_samples.append(
                    {
                        "audio": combined_audio,
                        "transcription": combined_transcription,
                        "filenames": combined_filenames,
                        "duration": current_duration,
                    }
                )
            else:
                sample = current_batch[0]
                combined_samples.append(
                    {
                        "audio": sample["audio"],
                        "transcription": sample["transcription"],
                        "filenames": [sample["filename"]],
                        "duration": sample["duration"],
                    }
                )

        logger.info(
            f"Combined {len(samples)} samples to {len(combined_samples)} batches"
        )
        return combined_samples

    def _split_by_duration(self, samples: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        """Split dataset by audio length (not count) to maintain ~20% validation split."""
        logger.info("Splitting dataset by audio length...")

        # Sort by duration
        sorted_samples = sorted(samples, key=lambda x: x["duration"])

        total_duration = sum(s["duration"] for s in sorted_samples)
        target_val_duration = total_duration * VALIDATION_SPLIT

        train_samples = []
        val_samples = []
        val_duration = 0.0

        # Go through samples and assign to validation until we hit target
        for sample in reversed(sorted_samples):  # Start from shortest for diversity
            if val_duration < target_val_duration:
                val_samples.append(sample)
                val_duration += sample["duration"]
            else:
                train_samples.append(sample)

        logger.info(
            f"Train set: {len(train_samples)} samples ({sum(s['duration'] for s in train_samples):.1f}s)"
        )
        logger.info(
            f"Val set: {len(val_samples)} samples ({sum(s['duration'] for s in val_samples):.1f}s)"
        )

        return train_samples, val_samples

    def load(self) -> Tuple[List[AudioSample], List[AudioSample]]:
        """Load and preprocess all data. Returns (train_samples, val_samples)."""
        # Try to load from cache first
        train_samples, val_samples = self._load_cache()
        if train_samples is not None and val_samples is not None:
            return train_samples, val_samples
        
        # Cache miss or disabled - process audio from scratch
        logger.info("Processing audio data from scratch...")
        
        # Load all audio files from all subfolders
        all_samples = self._discover_and_load_samples()

        if not all_samples:
            logger.error("No audio samples loaded!")
            return [], []

        logger.info(f"Total samples before combining: {len(all_samples)}")

        # Combine short audios
        combined_samples = self._combine_short_audios(all_samples)

        # Split by duration
        train_samples_raw, val_samples_raw = self._split_by_duration(combined_samples)

        # Convert to AudioSample objects
        train_samples = [
            AudioSample(
                audio_data=s["audio"],
                transcription=s["transcription"],
                original_filenames=s["filenames"],
                duration=s["duration"],
            )
            for s in train_samples_raw
        ]

        val_samples = [
            AudioSample(
                audio_data=s["audio"],
                transcription=s["transcription"],
                original_filenames=s["filenames"],
                duration=s["duration"],
            )
            for s in val_samples_raw
        ]
        
        # Save to cache for future runs
        if self.use_cache:
            self._save_cache(train_samples, val_samples)

        return train_samples, val_samples
