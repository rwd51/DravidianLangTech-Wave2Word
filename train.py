"""
Main Training Script for Tamil Dialect Classification using Whisper
"""
import os
import gc
import json
import random
import numpy as np
import torch
import evaluate

from transformers import (
    WhisperProcessor,
    WhisperFeatureExtractor,
    WhisperTokenizer,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments
)

from config import (
    MODEL_NAME,
    LANGUAGE,
    TASK,
    SEED,
    OUTPUT_DIR,
    TRAIN_DIR,
    DIALECT_DIRS,
    DIALECT_TO_LABEL,
    LABEL_TO_DIALECT,
    BATCH_SIZE,
    GRADIENT_ACCUMULATION_STEPS,
    LEARNING_RATE,
    NUM_EPOCHS,
    WEIGHT_DECAY,
    WARMUP_STEPS,
    LOGGING_STEPS,
    FP16,
    MAX_LENGTH,
    AUGMENT_TRAIN,
    AUGMENT_VAL,
    NUM_REGIONS,
    ADAPTER_DIM
)

from data_loader import (
    load_dialect_data,
    create_train_val_split,
    save_val_split_info
)
from dataset import TamilDialectDataset
from model import RegionalAdapterWhisper
from data_collator import DataCollatorRegionalASR
from trainer import RegionalTrainer, compute_metrics_factory


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    """Main training function."""
    print("=" * 80)
    print("Tamil Dialect Classification Training")
    print("=" * 80)

    # Set random seed
    set_seed(SEED)
    print(f"\nRandom seed set to: {SEED}")

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    # =========================================================================
    # Load Model and Processor
    # =========================================================================
    print("\n" + "=" * 80)
    print("Loading Whisper Model and Processor")
    print("=" * 80)

    print(f"\nModel: {MODEL_NAME}")
    print(f"Language: {LANGUAGE}")
    print(f"Task: {TASK}")

    # Load processor components
    feature_extractor = WhisperFeatureExtractor.from_pretrained(MODEL_NAME)
    tokenizer = WhisperTokenizer.from_pretrained(
        MODEL_NAME, language=LANGUAGE, task=TASK
    )
    processor = WhisperProcessor.from_pretrained(
        MODEL_NAME, language=LANGUAGE, task=TASK
    )

    # Load base Whisper model
    base_model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME)
    print(f"Base model loaded with {sum(p.numel() for p in base_model.parameters()):,} parameters")

    # Create Regional Adapter model
    regional_model = RegionalAdapterWhisper(
        original_whisper=base_model,
        num_regions=NUM_REGIONS,
        adapter_dim=ADAPTER_DIM
    )

    # Move to device
    regional_model = regional_model.to(device)
    print(f"\nRegional model created and moved to {device}")

    # =========================================================================
    # Load and Prepare Data
    # =========================================================================
    print("\n" + "=" * 80)
    print("Loading Dataset")
    print("=" * 80)

    # Load all training data
    audio_paths, transcriptions, dialects = load_dialect_data(
        TRAIN_DIR,
        DIALECT_DIRS
    )

    # Create train/val split (reproducible with fixed seed)
    (train_audio, train_trans, train_dialects,
     val_audio, val_trans, val_dialects) = create_train_val_split(
        audio_paths, transcriptions, dialects,
        seed=SEED
    )

    # Save validation split info for reproducibility
    val_split_path = os.path.join(OUTPUT_DIR, "val_split.json")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    save_val_split_info(val_audio, val_trans, val_dialects, val_split_path)

    # Create datasets
    print("\nCreating PyTorch datasets...")

    train_dataset = TamilDialectDataset(
        audio_files=train_audio,
        transcriptions=train_trans,
        dialects=train_dialects,
        processor=processor,
        dialect_to_idx=DIALECT_TO_LABEL,
        augment=AUGMENT_TRAIN
    )

    val_dataset = TamilDialectDataset(
        audio_files=val_audio,
        transcriptions=val_trans,
        dialects=val_dialects,
        processor=processor,
        dialect_to_idx=DIALECT_TO_LABEL,
        augment=AUGMENT_VAL
    )

    print(f"Train dataset: {len(train_dataset)} samples (augmentation: {AUGMENT_TRAIN})")
    print(f"Validation dataset: {len(val_dataset)} samples (augmentation: {AUGMENT_VAL})")

    # =========================================================================
    # Create Data Collator
    # =========================================================================
    print("\n" + "=" * 80)
    print("Creating Data Collator")
    print("=" * 80)

    decoder_start_token_id = base_model.config.decoder_start_token_id
    data_collator = DataCollatorRegionalASR(
        processor=processor,
        decoder_start_token_id=decoder_start_token_id
    )
    print("Regional data collator created")

    # =========================================================================
    # Setup Training Arguments
    # =========================================================================
    print("\n" + "=" * 80)
    print("Configuring Training")
    print("=" * 80)

    training_args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,

        # Evaluation strategy
        eval_strategy="epoch",
        save_strategy="epoch",

        logging_steps=LOGGING_STEPS,

        learning_rate=LEARNING_RATE,
        num_train_epochs=NUM_EPOCHS,
        weight_decay=WEIGHT_DECAY,
        warmup_steps=WARMUP_STEPS,

        save_total_limit=2,  # Keep best 2 checkpoints
        fp16=FP16 and torch.cuda.is_available(),
        predict_with_generate=True,

        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,

        gradient_checkpointing=False,
        generation_max_length=MAX_LENGTH,

        # Disable wandb
        report_to="none",

        # Remove unused columns (we handle this in trainer)
        remove_unused_columns=False,

        # Use pytorch format instead of safetensors (Whisper has tied weights)
        save_safetensors=False,
    )

    print(f"Training batch size: {BATCH_SIZE}")
    print(f"Gradient accumulation steps: {GRADIENT_ACCUMULATION_STEPS}")
    print(f"Effective batch size: {BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Number of epochs: {NUM_EPOCHS}")
    print(f"FP16 training: {training_args.fp16}")

    # =========================================================================
    # Setup Metrics
    # =========================================================================
    print("\n" + "=" * 80)
    print("Setting up Metrics")
    print("=" * 80)

    wer_metric = evaluate.load("wer")
    compute_metrics = compute_metrics_factory(processor, wer_metric)
    print("WER metric loaded")

    # =========================================================================
    # Initialize Trainer
    # =========================================================================
    print("\n" + "=" * 80)
    print("Initializing Trainer")
    print("=" * 80)

    trainer = RegionalTrainer(
        regional_model=regional_model,
        model=regional_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=processor,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    print("Regional trainer initialized")
    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Validation samples: {len(val_dataset)}")

    # =========================================================================
    # Train
    # =========================================================================
    print("\n" + "=" * 80)
    print("Starting Training")
    print("=" * 80)

    # Clear GPU memory before training
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    # Train the model
    train_result = trainer.train()

    # =========================================================================
    # Save Final Model
    # =========================================================================
    print("\n" + "=" * 80)
    print("Saving Model")
    print("=" * 80)

    # Save the trainer state
    trainer.save_model(OUTPUT_DIR)
    trainer.save_state()

    # Save regional model separately
    regional_model.save_pretrained(os.path.join(OUTPUT_DIR, "regional_adapter"))

    # Save training metrics
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)

    # Save dialect mapping
    dialect_mapping = {
        "dialect_to_label": DIALECT_TO_LABEL,
        "label_to_dialect": LABEL_TO_DIALECT,
        "num_regions": NUM_REGIONS
    }
    with open(os.path.join(OUTPUT_DIR, "dialect_mapping.json"), 'w') as f:
        json.dump(dialect_mapping, f, indent=2)

    print(f"\nModel and training artifacts saved to: {OUTPUT_DIR}")

    # =========================================================================
    # Final Evaluation
    # =========================================================================
    print("\n" + "=" * 80)
    print("Final Evaluation on Validation Set")
    print("=" * 80)

    eval_results = trainer.evaluate()
    trainer.log_metrics("eval", eval_results)
    trainer.save_metrics("eval", eval_results)

    print(f"\nFinal validation WER: {eval_results.get('eval_wer', 'N/A'):.2f}%")

    print("\n" + "=" * 80)
    print("Training Complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
