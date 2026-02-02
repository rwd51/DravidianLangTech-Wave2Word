"""
Configuration file for Tamil Dialect Classification using Whisper
"""
import os

# =============================================================================
# Paths Configuration
# =============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = "/kaggle/input/acl2026-tamildialectclassification-og-1/Dialect_Based_Speech_Recognition-20260201T123636Z-3-001/Dialect_Based_Speech_Recognition"
TRAIN_DIR = "/kaggle/input/acl2026-tamildialectclassification-og-1/Dialect_Based_Speech_Recognition-20260201T123636Z-3-001/Dialect_Based_Speech_Recognition/Train"
TEST_DIR = "/kaggle/input/acl2026-tamildialectclassification-og-1/Dialect_Based_Speech_Recognition-20260201T123636Z-3-001/Dialect_Based_Speech_Recognition/Test"
OUTPUT_DIR ="/kaggle/working/tamil-dialect-whisper-finetuned"

# Dialect directories
DIALECT_DIRS = {
    "Northern_Dialect": "Northern_Dialect",
    "Southern_Dialect": "Southern_Dialect",
    "Western_Dialect": "Western_Dialect",
    "Central_Dialect": "Central_Dialect"
}

# =============================================================================
# Model Configuration
# =============================================================================
MODEL_NAME = "vasista22/whisper-tamil-medium"
LANGUAGE = "Tamil"
TASK = "transcribe"

# =============================================================================
# Training Hyperparameters
# =============================================================================
SEED = 42
SAMPLING_RATE = 16000
MAX_LENGTH = 225

# Training settings
BATCH_SIZE = 64
GRADIENT_ACCUMULATION_STEPS = 16
EFFECTIVE_BATCH_SIZE = BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS

LEARNING_RATE = 1e-5
NUM_EPOCHS = 20
WEIGHT_DECAY = 0.01
WARMUP_STEPS = 500

# Validation split
VALIDATION_SPLIT = 0.10
TRAIN_SPLIT = 0.90

# Model architecture
ADAPTER_DIM = 64
NUM_REGIONS = 4  # Northern, Southern, Western, Central

# Loss weighting
ALPHA_REGION_LOSS = 0.3  # Weight for regional classification loss

# =============================================================================
# Augmentation settings
# =============================================================================
AUGMENT_TRAIN = True
AUGMENT_VAL = False

# Augmentation probabilities
TIME_STRETCH_PROB = 0.3
PITCH_SHIFT_PROB = 0.3
NOISE_PROB = 0.3
VOLUME_SHIFT_PROB = 0.3
RANDOM_CROP_PROB = 0.2

# =============================================================================
# Dialect to Label Mapping
# =============================================================================
DIALECT_TO_LABEL = {
    "Northern_Dialect": 0,
    "Southern_Dialect": 1,
    "Western_Dialect": 2,
    "Central_Dialect": 3
}

LABEL_TO_DIALECT = {v: k for k, v in DIALECT_TO_LABEL.items()}

# Display names for dialects
DIALECT_DISPLAY_NAMES = {
    "Northern_Dialect": "Northern dialect",
    "Southern_Dialect": "Southern dialect",
    "Western_Dialect": "Western dialect",
    "Central_Dialect": "Central dialect"
}

# =============================================================================
# Logging
# =============================================================================
LOGGING_STEPS = 200
SAVE_STRATEGY = "epoch"
EVAL_STRATEGY = "epoch"
FP16 = True
