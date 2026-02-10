# Fine-tuning configuration for Dolphin

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional

# ===================== USER CONFIGURATION =====================
# Modify these variables to customize the fine-tuning process

# Dataset Configuration
DATASET_ROOT = "/kaggle/input/datasets/shadabtanjeed/acl2026-tamildialectclassification-og-2/Train"  # Root folder containing audio subfolders (e.g., sp1_tha_audio, sp2_tha_audio, etc.)
# The loader will automatically discover and load all audio folders (*_audio) and their corresponding text files

# Pretrained Model Configuration
PRETRAINED_MODEL_DIR = "/kaggle/hf-dolphin-small"  # Directory with pretrained model (bpe.model, config.yaml, feats_stats.npz, small.pt)
OUTPUT_DIR = "/kaggle/dolphin-finetuned-output"  # Where to save the fine-tuned model

# Audio Processing Configuration
TARGET_LANGUAGE = "ta"  # Tamil language code
TARGET_REGION = "ta-IN"  # Tamil India region code
SAMPLE_RATE = 16000  # Sample rate in Hz
AUDIO_MIN_DURATION = 0.5  # Minimum audio duration in seconds
MAX_COMBINED_AUDIO_LENGTH = 28.0  # Maximum combined audio length in seconds

# Training Configuration
LEARNING_RATES = [1e-5, 5e-5]  # Learning rate range for schedule
OPTIMIZER = "adamw"  # Optimizer type
WARMUP_STEPS = 200  # Warmup steps
CTC_WEIGHT = 0.3  # CTC loss weight
DROPOUT_RANGE = [0.05, 0.1]  # Dropout range
NUM_EPOCHS = 4  # Number of training epochs
BATCH_SIZE = 8  # Batch size
GRADIENT_ACCUMULATION_STEPS = 2  # Gradient accumulation steps
MAX_GRAD_NORM = 1.0  # Maximum gradient norm for clipping

# Validation Configuration
VALIDATION_SPLIT = 0.2  # 20% validation split based on audio length
VAL_CHECK_INTERVAL = 500  # Validate every N steps

# Checkpoint Configuration
SAVE_ONLY_LATEST = True  # Save only the latest checkpoint
CHECKPOINT_DIR = "/kaggle/checkpoints/finetuned"  # Directory to save checkpoints

# Cache Configuration
CACHE_DIR = "/kaggle/working/audio_cache"  # Directory to cache processed audio data
USE_CACHE = True  # Whether to use cached audio data (set False to force reprocessing)

# Output Configuration
TRANSCRIPTION_OUTPUT = "/kaggle/output/transcriptions.txt"  # Output file for transcriptions
LOG_INTERVAL = 50  # Log metrics every N steps

# Device Configuration
DEVICE = "cuda"  # Device to use: 'cuda' or 'cpu'
DTYPE = "float32"  # Data type: 'float32' or 'float16'
NUM_WORKERS = 8  # Number of workers for DataLoader

# Random Seed
SEED = 42

# ===================== DERIVED CONFIGURATION =====================
# These are automatically computed from the above settings

# Create output directories
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(os.path.dirname(TRANSCRIPTION_OUTPUT), exist_ok=True)
if USE_CACHE:
    os.makedirs(CACHE_DIR, exist_ok=True)

# Special tokens
SPECIAL_TOKENS = {
    "sos": "<SOS>",
    "language": "<LANGUAGE>",
    "region": "<REGION>",
    "transcribe": "<TRANSCRIBE>",
    "eos": "<EOS>",
}

# Task token
TASK_TOKEN = "<asr>"


# Format template for combining tokens
def format_token_sequence(
    transcription: str, language: str = TARGET_LANGUAGE, region: str = TARGET_REGION
) -> str:
    """
    Format transcription with special tokens.
    Format: <SOS><LANGUAGE><ta><REGION><ta-IN><TRANSCRIBE>text<EOS>
    """
    return f"{SPECIAL_TOKENS['sos']}{SPECIAL_TOKENS['language']}<{language}>{SPECIAL_TOKENS['region']}<{region}>{SPECIAL_TOKENS['transcribe']}{transcription}{SPECIAL_TOKENS['eos']}"
