"""
Data Loader for Tamil Dialect Dataset
Parses the directory structure and creates train/val splits
"""
import os
import glob
from collections import defaultdict
from typing import Tuple, List, Dict
import soundfile as sf
import numpy as np
from sklearn.model_selection import train_test_split

from config import (
    TRAIN_DIR,
    TEST_DIR,
    DIALECT_DIRS,
    DIALECT_TO_LABEL,
    SEED,
    VALIDATION_SPLIT
)
from tamil_text_normalizer import create_normalizer


def parse_text_file(text_file_path: str) -> Dict[str, str]:
    """
    Parse a speaker's text file to get audio filename -> transcription mapping.

    Format in text file:
    SP32_KG_M_1 இன்னைக்கு என்னங்க ஒரே உப்பசமா இருக்குதுங்க.
    SP32_KG_M_2 அங்க என்ன பண்றிங்க வாங்க விஷ்க்குனு போலாம்.

    Args:
        text_file_path: Path to the text file

    Returns:
        Dictionary mapping audio filename (without extension) to transcription
    """
    transcriptions = {}

    try:
        with open(text_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                # Split on first space to separate filename from transcription
                parts = line.split(' ', 1)
                if len(parts) == 2:
                    audio_name = parts[0].strip()
                    transcription = parts[1].strip()
                    transcriptions[audio_name] = transcription
    except Exception as e:
        print(f"Error reading {text_file_path}: {e}")

    return transcriptions


def load_dialect_data(
    train_dir: str,
    dialect_dirs: Dict[str, str],
    normalizer=None
) -> Tuple[List[str], List[str], List[str]]:
    """
    Load all data from the training directory.

    Args:
        train_dir: Path to the Train directory
        dialect_dirs: Mapping of dialect names to directory names
        normalizer: Optional text normalizer

    Returns:
        Tuple of (audio_paths, transcriptions, dialects)
    """
    audio_paths = []
    transcriptions = []
    dialects = []

    if normalizer is None:
        normalizer = create_normalizer("default")

    for dialect_name, dialect_folder in dialect_dirs.items():
        dialect_path = os.path.join(train_dir, dialect_folder)

        if not os.path.exists(dialect_path):
            print(f"Warning: Dialect folder not found: {dialect_path}")
            continue

        print(f"\nProcessing {dialect_name}...")

        # Find all text files in this dialect folder
        text_files = glob.glob(os.path.join(dialect_path, "*.txt"))

        dialect_audio_count = 0
        dialect_trans_count = 0

        for text_file in text_files:
            # Parse the text file to get transcriptions
            file_transcriptions = parse_text_file(text_file)
            dialect_trans_count += len(file_transcriptions)

            # Get the speaker prefix from text filename
            # e.g., SP32_KG_Text.txt -> SP32_KG
            text_basename = os.path.basename(text_file)
            speaker_prefix = text_basename.replace("_Text.txt", "")

            # Find the corresponding audio folder
            audio_folder = os.path.join(dialect_path, f"{speaker_prefix}_audio")

            if not os.path.exists(audio_folder):
                # Try alternate naming
                audio_folder = os.path.join(dialect_path, speaker_prefix + "_audio")

            if os.path.exists(audio_folder):
                # Get all wav files in the audio folder
                wav_files = glob.glob(os.path.join(audio_folder, "*.wav"))
                wav_files.extend(glob.glob(os.path.join(audio_folder, "*.WAV")))

                for wav_file in wav_files:
                    wav_basename = os.path.basename(wav_file)
                    audio_name = os.path.splitext(wav_basename)[0]

                    # Check if we have a transcription for this audio
                    if audio_name in file_transcriptions:
                        transcription = file_transcriptions[audio_name]

                        # Apply text normalization
                        normalized_transcription = normalizer(transcription)

                        if normalized_transcription:  # Skip empty transcriptions
                            audio_paths.append(wav_file)
                            transcriptions.append(normalized_transcription)
                            dialects.append(dialect_name)
                            dialect_audio_count += 1

        print(f"  Found {dialect_audio_count} audio files with transcriptions")
        print(f"  (from {len(text_files)} text files with {dialect_trans_count} entries)")

    print(f"\nTotal: {len(audio_paths)} samples loaded")
    return audio_paths, transcriptions, dialects


def get_audio_duration(audio_path: str) -> float:
    """
    Get the duration of an audio file in seconds.

    Args:
        audio_path: Path to audio file

    Returns:
        Duration in seconds
    """
    try:
        info = sf.info(audio_path)
        return info.duration
    except Exception:
        return 0.0


def create_train_val_split(
    audio_paths: List[str],
    transcriptions: List[str],
    dialects: List[str],
    val_split: float = VALIDATION_SPLIT,
    seed: int = SEED,
    stratify_by_duration: bool = True
) -> Tuple[List, List, List, List, List, List]:
    """
    Create stratified train/validation split maintaining dialect ratios.

    Args:
        audio_paths: List of audio file paths
        transcriptions: List of transcriptions
        dialects: List of dialect labels
        val_split: Validation split ratio
        seed: Random seed for reproducibility
        stratify_by_duration: Whether to stratify by audio duration bins

    Returns:
        Tuple of (train_audio, train_trans, train_dialects,
                  val_audio, val_trans, val_dialects)
    """
    train_audio_paths = []
    train_transcriptions = []
    train_dialects = []

    val_audio_paths = []
    val_transcriptions = []
    val_dialects = []

    print("\nCreating stratified train/validation split by dialect...")
    print(f"Target split: {(1-val_split)*100:.0f}% train, {val_split*100:.0f}% validation")

    # Group data by dialect
    dialect_data = defaultdict(list)
    for i, (audio, trans, dialect) in enumerate(zip(audio_paths, transcriptions, dialects)):
        dialect_data[dialect].append((audio, trans))

    # Split each dialect separately to maintain ratios
    for dialect, data in dialect_data.items():
        dialect_audio = [d[0] for d in data]
        dialect_trans = [d[1] for d in data]

        if len(dialect_audio) < 2:
            # If only 1 sample, put in training
            train_audio_paths.extend(dialect_audio)
            train_transcriptions.extend(dialect_trans)
            train_dialects.extend([dialect] * len(dialect_audio))
            print(f"  {dialect}: {len(dialect_audio)} sample(s) -> all to train (too few)")
            continue

        # Get duration bins for stratification if enabled
        stratify_labels = None
        if stratify_by_duration and len(dialect_audio) >= 10:
            try:
                durations = [get_audio_duration(a) for a in dialect_audio]
                # Create 3 bins based on duration
                bins = np.percentile(durations, [33, 66])
                stratify_labels = np.digitize(durations, bins)
            except Exception:
                stratify_labels = None

        try:
            train_paths, val_paths, train_texts, val_texts = train_test_split(
                dialect_audio,
                dialect_trans,
                test_size=val_split,
                random_state=seed,
                shuffle=True,
                stratify=stratify_labels
            )
        except ValueError:
            # Fall back to simple split if stratification fails
            train_paths, val_paths, train_texts, val_texts = train_test_split(
                dialect_audio,
                dialect_trans,
                test_size=val_split,
                random_state=seed,
                shuffle=True
            )

        train_audio_paths.extend(train_paths)
        train_transcriptions.extend(train_texts)
        train_dialects.extend([dialect] * len(train_paths))

        val_audio_paths.extend(val_paths)
        val_transcriptions.extend(val_texts)
        val_dialects.extend([dialect] * len(val_paths))

        print(f"  {dialect}: {len(train_paths)} train, {len(val_paths)} val "
              f"({len(val_paths)/len(dialect_audio)*100:.1f}%)")

    print(f"\nTotal: {len(train_audio_paths)} train, {len(val_audio_paths)} val")
    print(f"Split ratio: {len(train_audio_paths)/(len(train_audio_paths)+len(val_audio_paths))*100:.1f}% / "
          f"{len(val_audio_paths)/(len(train_audio_paths)+len(val_audio_paths))*100:.1f}%")

    return (
        train_audio_paths, train_transcriptions, train_dialects,
        val_audio_paths, val_transcriptions, val_dialects
    )


def load_test_data(test_dir: str) -> List[str]:
    """
    Load test audio file paths.

    Args:
        test_dir: Path to test directory

    Returns:
        List of test audio file paths
    """
    test_files = glob.glob(os.path.join(test_dir, "*.wav"))
    test_files.extend(glob.glob(os.path.join(test_dir, "*.WAV")))

    # Sort for consistent ordering
    test_files = sorted(test_files)

    print(f"Found {len(test_files)} test audio files")
    return test_files


def save_val_split_info(
    val_audio_paths: List[str],
    val_transcriptions: List[str],
    val_dialects: List[str],
    output_path: str
):
    """
    Save validation split information for reproducibility.

    Args:
        val_audio_paths: Validation audio paths
        val_transcriptions: Validation transcriptions
        val_dialects: Validation dialect labels
        output_path: Path to save the info
    """
    import json

    val_info = {
        "num_samples": len(val_audio_paths),
        "samples": [
            {
                "audio_path": audio,
                "transcription": trans,
                "dialect": dialect
            }
            for audio, trans, dialect in zip(
                val_audio_paths, val_transcriptions, val_dialects
            )
        ]
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(val_info, f, ensure_ascii=False, indent=2)

    print(f"Validation split info saved to {output_path}")


def load_val_split_info(input_path: str) -> Tuple[List[str], List[str], List[str]]:
    """
    Load validation split information.

    Args:
        input_path: Path to the saved info

    Returns:
        Tuple of (val_audio_paths, val_transcriptions, val_dialects)
    """
    import json

    with open(input_path, 'r', encoding='utf-8') as f:
        val_info = json.load(f)

    val_audio_paths = [s["audio_path"] for s in val_info["samples"]]
    val_transcriptions = [s["transcription"] for s in val_info["samples"]]
    val_dialects = [s["dialect"] for s in val_info["samples"]]

    print(f"Loaded {len(val_audio_paths)} validation samples from {input_path}")
    return val_audio_paths, val_transcriptions, val_dialects


if __name__ == "__main__":
    # Test the data loader
    print("Testing Tamil Dialect Data Loader")
    print("=" * 60)

    # Load all data
    audio_paths, transcriptions, dialects = load_dialect_data(
        TRAIN_DIR,
        DIALECT_DIRS
    )

    # Create train/val split
    (train_audio, train_trans, train_dialects,
     val_audio, val_trans, val_dialects) = create_train_val_split(
        audio_paths, transcriptions, dialects
    )

    # Show sample data
    print("\n" + "=" * 60)
    print("Sample data from validation set:")
    for i in range(min(3, len(val_audio))):
        print(f"\nSample {i+1}:")
        print(f"  Audio: {os.path.basename(val_audio[i])}")
        print(f"  Dialect: {val_dialects[i]}")
        print(f"  Transcription: {val_trans[i][:50]}...")
