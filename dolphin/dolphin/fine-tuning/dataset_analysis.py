#!/usr/bin/env python3
"""
Dataset preparation and analysis utility.
Helps organize and validate your dataset before training.
"""

import os
import logging
from pathlib import Path
from typing import Dict, List
import numpy as np
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dolphin.audio import load_audio
from finetune_config import DATASET_ROOT, SAMPLE_RATE

logging.basicConfig(
    level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


class DatasetAnalyzer:
    """Analyze dataset structure and statistics."""

    def __init__(self, dataset_root: str):
        """Initialize analyzer."""
        self.dataset_root = Path(dataset_root)
        self.stats = {}

    def _discover_audio_folders(self) -> List[Path]:
        """Discover all audio folders in dataset root."""
        return sorted([d for d in self.dataset_root.rglob("*_audio") if d.is_dir()])
    
    def analyze_folder(self, audio_folder: Path) -> Dict:
        """Analyze a single audio folder."""
        if not audio_folder.exists():
            logger.warning(f"Folder not found: {audio_folder}")
            "text_files": 0,
            "total_audios": 0,
            "total_duration": 0.0,
            "min_duration": float("inf"),
            "max_duration": 0.0,
            "issues": [],
        

        # Find audio folders and text files
        audio_folders = sorted(
            [
                d
                for d in dialect_path.iterdir()
                if d.is_dir() and d.name.endswith("_audio")
            ]
        )
        text_files = sorted(dialect_path.glob("*_Text.txt"))

        stats["audio_folders"] = len(audio_folders)
        stats["text_files"] = len(text_files)

        logger.info(f"  Audio folders: {len(audio_folders)}")
        logger.info(f"  Text files: {len(text_files)}")

        # Analyze each speaker
        for audio_folder in audio_folders:
            base_name = audio_folder.name.replace("_audio", "")
            text_file = dialect_path / f"{base_name}_Text.txt"

            # Load transcriptions
            transcriptions = {}
            if text_file.exists():
                with open(text_file, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            parts = line.split(None, 1)
                            if len(parts) == 2:
                                transcriptions[parts[0]] = parts[1]
            else:
                stats["issues"].append(f"Text file missing: {text_file.name}")

            # Process audio files
            wav_files = sorted(audio_folder.glob("*.wav"))

            for wav_file in wav_files:
            filename = wav_file.stem
            
            try:
                # Load audio and get duration
                audio_data = load_audio(str(wav_file), sr=SAMPLE_RATE)
                duration = len(audio_data) / SAMPLE_RATE
                
                stats["total_audios"] += 1
                stats["total_duration"] += duration
                stats["min_duration"] = min(stats["min_duration"], duration)
                stats["max_duration"] = max(stats["max_duration"], duration)
                
                # Check if transcription exists
                if filename not in transcriptions:
                    stats["issues"].append(f"Missing transcription for {filename}")
                
            except Exception as e:
                stats["issues"].append(f"Error loading {wav_file.name}: {e}")
        # Log summary
        if stats["total_audios"] > 0:
            logger.info(f"  Total audios: {stats['total_audios']}")
            logger.info(
                f"  Total duration: {stats['total_duration']:.1f}s ({stats['total_duration']/60:.1f}m)"
            )
            logger.info(
                f"  Avg duration: {stats['total_duration']/stats['total_audios']:.2f}s"
            )
            logger.info(f"  Min duration: {stats['min_duration']:.2f}s")
            logger.info(f"  Max duration: {stats['max_duration']:.2f}s")

        if stats["issues"]:
            logger.warning(f"  Issues found: {len(stats['issues'])}")
            for issue in stats["issues"][:5]:  # Show first 5
                logger.warning(f"    - {issue}")
            if len(stats["issues"]) > 5:
                logger.warning(f"    ... and {len(stats['issues']) - 5} more")

        return stats

def analyze_all(self) -> Dict:
        """Analyze all audio folders in dataset."""
        logger.info("=" * 80)
        logger.info("DATASET ANALYSIS")
        logger.info("=" * 80)
        
        audio_folders = self._discover_audio_folders()
        
        if not audio_folders:
            logger.warning(f"No audio folders found in {self.dataset_root}")
            return {}
        
        all_stats = {}
        total_audios = 0
        total_duration = 0.0
        
        for audio_folder in audio_folders:
            stats = self.analyze_folder(audio_folder)
            if stats:
                all_stats[audio_folder.name] = stats
                total_audios += stats["total_audios"]
                total_duration += stats["total_duration"]

        # Overall summary
        logger.info("\n" + "=" * 80)
        logger.info("OVERALL SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Total dialects: {len(all_stats)}")
        logger.info(f"Total audios: {total_audios}")
        logger.info(
            f"Total duration: {total_duration:.1f}s ({total_duration/3600:.1f}h)"
        )
        if total_audios > 0:
            logger.info(f"Avg audio duration: {total_duration/total_audios:.2f}s")

        # Dataset size after combining
        if total_audios > 0:
            logger.info(f"\nAfter combining audios (max 28s each):")
            estimated_combined = total_duration / 28.0
            logger.info(f"Estimated samples: ~{int(estimated_combined)}")
            logger.info(f"Train set (80%): ~{int(estimated_combined * 0.8)}")
            logger.info(f"Val set (20%): ~{int(estimated_combined * 0.2)}")

        return all_stats


class DatasetFormatter:
    """Helper to fix common dataset format issues."""

    @staticmethod
    def check_text_file_encoding(text_file: Path) -> bool:
        """Check if text file is valid UTF-8."""
        try:
            with open(text_file, "r", encoding="utf-8") as f:
                f.read()
            return True
        except UnicodeDecodeError:
            return False

    @staticmethod
    def fix_text_file_encoding(text_file: Path):
        """Try to fix text file encoding."""
        logger.info(f"Fixing encoding for {text_file.name}...")

        try:
            # Try different encodings
            encodings = ["utf-16", "cp1252", "latin-1"]

            for encoding in encodings:
                try:
                    with open(text_file, "r", encoding=encoding) as f:
                        content = f.read()

                    # Save as UTF-8
                    with open(text_file, "w", encoding="utf-8") as f:
                        f.write(content)

                    logger.info(f"  ✓ Fixed using {encoding} encoding")
                    return True
                except:
                    continue

            logger.warning(f"  ✗ Could not fix encoding for {text_file.name}")
            return False

        except Exception as e:
            logger.error(f"  Error fixing {text_file.name}: {e}")
            return False

    @staticmethod
    def check_transcription_format(text_file: Path) -> bool:
        """Check if transcription file has correct format."""
        try:
            with open(text_file, "r", encoding="utf-8") as f:
                lines = f.readlines()

            issues = []
            for i, line in enumerate(lines, 1):
                line = line.strip()
                if not line:
                    continue

                parts = line.split(None, 1)
                if len(parts) < 2:
                    issues.append(f"Line {i}: Missing transcription - {line[:50]}")
                elif " " not in line:
                    issues.append(
                        f"Line {i}: No space between filename and transcription"
                    )

            return len(issues) == 0, issues

        except Exception as e:
            return False, [str(e)]


def main():
    """Run dataset analysis."""
    analyzer = DatasetAnalyzer(DATASET_ROOT)
    stats = analyzer.analyze_all()
    
    # Check for encoding issues
    logger.info("\n" + "=" * 80)
    logger.info("CHECKING TEXT FILE ENCODING")
    logger.info("=" * 80)
    
    audio_folders = analyzer._discover_audio_folders()
    text_files = sorted(Path(DATASET_ROOT).rglob("*_Text.txt"))
    
    for text_file in text_files:
        formatter = DatasetFormatter()
        
        if not formatter.check_text_file_encoding(text_file):
            logger.warning(f"Encoding issue: {text_file.name}")
            # Uncomment to auto-fix:
            # formatter.fix_text_file_encoding(text_file)
        
        ok, issues = formatter.check_transcription_format(text_file)
        if not ok:
            logger.warning(f"Format issues in {text_file.name}:")
            for issue in issues[:3]:
                logger.warning(f"  {issue}")
    logger.info("Ready to train? Run: python finetune.py")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
