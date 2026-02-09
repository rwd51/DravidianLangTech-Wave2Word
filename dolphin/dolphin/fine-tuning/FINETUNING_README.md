# Fine-tuning Dolphin Speech Recognition for Tamil Dialects

This directory contains a complete fine-tuning pipeline for the Dolphin speech recognition model, optimized for Tamil language with support for multiple dialects.

## Project Structure

```
fine-tuning/
├── finetune_config.py          # Configuration - EDIT THIS FOR YOUR SETUP
├── finetune_dataset.py          # Dataset loading and preprocessing
├── finetune_utils.py            # Training utilities and metrics
├── finetune.py                  # Main training script
├── finetune_inference.py        # Inference and transcription utilities
├── load-dolphin-small-pretrained.py  # Original pretrained model loader
└── README.md                    # This file

data/
├── dialects/                    # Your dataset directory
│   ├── central_dialect/
│   │   ├── sp1_tha_audio/      # Audio files
│   │   ├── sp1_tha_Text.txt    # Transcriptions
│   │   ├── sp2_tha_audio/
│   │   ├── sp2_tha_Text.txt
│   │   └── ...
│   ├── southern_dialect/
│   ├── northern_dialect/
│   └── eastern_dialect/

dolphin/
└── assets/                      # Pretrained model assets
    ├── config.yaml
    ├── feats_stats.npz
    ├── bpe.model
    └── small.pt

output/
├── finetuned_model/            # Fine-tuned model checkpoint
└── transcriptions.txt           # Output transcriptions

checkpoints/
└── finetuned/                  # Training checkpoints
```

## Configuration

Before running training, configure the file `finetune_config.py`:

### Dataset Configuration

- `DATASET_ROOT`: Path to your dialect folders
- `DIALECTS`: List of dialect folder names to train on
- `AUDIO_MIN_DURATION`: Minimum audio duration to include (default: 0.5s)
- `MAX_COMBINED_AUDIO_LENGTH`: Maximum length for combining short audios (default: 28s)

### Model Configuration

- `PRETRAINED_MODEL_DIR`: Path to pretrained model assets
- `OUTPUT_DIR`: Where to save fine-tuned model
- `CHECKPOINT_DIR`: Where to save training checkpoints

### Training Hyperparameters

- `LEARNING_RATES`: [1e-5, 5e-5] - Learning rate range
- `WARMUP_STEPS`: 200
- `CTC_WEIGHT`: 0.3
- `DROPOUT_RANGE`: [0.05, 0.1]
- `NUM_EPOCHS`: 4
- `BATCH_SIZE`: 4
- `GRADIENT_ACCUMULATION_STEPS`: 2

### Validation

- `VALIDATION_SPLIT`: 0.2 (20% based on audio length, not count)
- `VAL_CHECK_INTERVAL`: 500 steps

### Special Tokens

The model uses special tokens in the format:

```
<SOS><LANGUAGE><ta><REGION><ta-IN><TRANSCRIBE>text<EOS>
```

## Dataset Format

Your dataset should follow this structure:

```
data/dialects/
└── central_dialect/
    ├── sp1_tha_audio/          # Folder with WAV files
    │   ├── file1.wav
    │   ├── file2.wav
    │   └── ...
    ├── sp1_tha_Text.txt        # Transcription file
    ├── sp2_tha_audio/
    ├── sp2_tha_Text.txt
    └── ...
```

**Transcription file format** (sp1_tha_Text.txt):

```
file1 Transcription for file 1
file2 Transcription for file 2
...
```

Note: Filenames in the text file should NOT include the .wav extension.

## Features

### Automatic Audio Combining

If audio files are shorter than 28 seconds, the script automatically combines multiple files:

- Combines audios while respecting MAX_COMBINED_AUDIO_LENGTH
- Concatenates transcriptions with spaces
- Performs this BEFORE train/val split

### Intelligent Validation Split

- Splits dataset by **audio length**, not by sample count
- Ensures 20% of total audio length goes to validation
- Maintains dialect diversity

### Special Tokens

All transcriptions are wrapped with special tokens:

- `<SOS>`: Start of Sequence
- `<LANGUAGE><ta>`: Language identifier
- `<REGION><ta-IN>`: Region identifier
- `<TRANSCRIBE>`: Task identifier
- `<EOS>`: End of Sequence

### Metrics and Logging

- **WER (Word Error Rate)**: Primary evaluation metric
- **Training Loss**: Monitored every LOG_INTERVAL steps
- **Progress Bars**: Using tqdm for all operations
- **Real-time Logging**: Console output shows loss, WER, learning rate

## Running Training

### Basic Usage

```bash
# Edit configuration first
vi finetune_config.py

# Run training
python finetune.py
```

### Training Output

During training, you'll see:

```
[timestamp] [INFO] Loading dataset...
[timestamp] [INFO] Combining short audio files...
[timestamp] [INFO] Splitting dataset by audio length...
[timestamp] [INFO] Train set: 150 samples (3245.2s)
[timestamp] [INFO] Val set: 40 samples (812.5s)

[timestamp] [INFO] Training epoch 1/4
Epoch 1: 100%|██████████| 150/150 [12:34<00:00, 5.02s/batch]
Step   100 | Train Loss: 2.3456 | LR: 1.25e-05
Step   200 | Train Loss: 1.9876 | Val Loss: 2.1234 | WER: 0.2345
```

### Checkpoints

- Training checkpoints are saved to `CHECKPOINT_DIR`
- Only the latest checkpoint is kept (controlled by `SAVE_ONLY_LATEST`)
- Each checkpoint contains:
  - Model weights
  - Optimizer state
  - Scheduler state
  - Training metrics

### Final Output

After training completes:

- Fine-tuned model saved to `OUTPUT_DIR`
- Transcriptions saved to `TRANSCRIPTION_OUTPUT`
- Model copies: config.yaml, bpe.model, feats_stats.npz, small.pt

## Using the Fine-tuned Model

### For Inference

The fine-tuned model can be loaded using the standard DolphinSpeech2Text loader:

```python
from dolphin.model import DolphinSpeech2Text

model = DolphinSpeech2Text(
    s2t_train_config="output/finetuned_model/config.yaml",
    s2t_model_file="output/finetuned_model/small.pt",
    device="cuda",
    beam_size=5
)

# Transcribe audio
results = model(audio_data, sr=16000)
print(results.text)
```

### Run Inference Only

```python
from finetune_inference import InferenceEngine

engine = InferenceEngine("output/finetuned_model")
transcription = engine.transcribe(audio_data)
print(transcription)
```

## Key Design Decisions

1. **Multiple Python Files**: Separated concerns for easier debugging and reuse
   - `finetune_config.py`: All configuration in one place
   - `finetune_dataset.py`: Data handling
   - `finetune_utils.py`: Training utilities
   - `finetune.py`: Main training loop

2. **Audio Combining Before Split**: Ensures validation set balance

3. **Length-based Validation Split**: Better than count-based for variable-length audio

4. **Single Checkpoint**: Reduces disk usage, keeps latest best model

5. **Special Tokens**: Enables multi-task learning (language, region, task explicit)

6. **Progress Bars**: tqdm at every step for transparency

7. **Linux Path Convention**: Uses forward slashes (/) for cross-platform compatibility

## Troubleshooting

### CUDA Out of Memory

- Reduce `BATCH_SIZE` in finetune_config.py
- Reduce `MAX_COMBINED_AUDIO_LENGTH`
- Use gradient accumulation

### Slow Training

- Reduce `NUM_WORKERS`
- Ensure dataset is on fast storage (SSD)
- Use multiple GPUs if available

### Missing Audio Files

- Check that WAV files exist in the dialect folders
- Verify transcription filenames match audio files (without .wav extension)
- Check encoding of text files (should be UTF-8)

### Model Not Learning

- Check learning rate (currently 1e-5 to 5e-5)
- Verify special tokens are formatted correctly
- Ensure audio quality is reasonable
- Check that transcriptions are not empty

## References

- **ESPnet**: https://github.com/espnet/espnet
- **Dolphin**: Original speech recognition model in this repository
- **WER Metric**: https://huggingface.co/spaces/evaluate-metric/wer

## Contact & Support

For issues with the fine-tuning pipeline, check:

1. Dataset format matches expected structure
2. All paths in finetune_config.py are correct
3. Required packages are installed (espnet, torch, tqdm, etc.)
4. Sufficient disk space for checkpoints and outputs
