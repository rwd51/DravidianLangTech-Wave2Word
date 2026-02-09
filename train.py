"""
Main Training Script for Tamil Dialect Classification using Whisper
"""
import os
import gc
import json
import random
import shutil
import numpy as np
import torch
import evaluate

from transformers import (
    WhisperProcessor,
    WhisperFeatureExtractor,
    WhisperTokenizer,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    TrainerCallback
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


def cleanup_checkpoints(output_dir):
    """Remove ALL checkpoint-* directories from output_dir."""
    if not os.path.exists(output_dir):
        return 0
    removed = 0
    for item in os.listdir(output_dir):
        if item.startswith("checkpoint-"):
            item_path = os.path.join(output_dir, item)
            if os.path.isdir(item_path):
                shutil.rmtree(item_path)
                print(f"Removed checkpoint: {item}")
                removed += 1
    return removed


class EpochUpdateCallback(TrainerCallback):
    """
    Callback to update dataset epoch for curriculum-based augmentation.
    Augmentation intensity increases as training progresses.
    """
    def __init__(self, train_dataset):
        self.train_dataset = train_dataset

    def on_epoch_begin(self, args, state, control, **kwargs):
        """Update dataset epoch at the start of each epoch."""
        epoch = int(state.epoch) if state.epoch else 0
        self.train_dataset.set_epoch(epoch)
        intensity = self.train_dataset.get_augmentation_intensity()
        print(f"\n[Curriculum] Epoch {epoch}: Augmentation intensity = {intensity:.2f}")


class SaveBestModelCallback(TrainerCallback):
    """
    Callback that saves ONLY the best model based on COMBINED score.
    Combined Score = (100 - WER) * 0.5 + Dialect_Accuracy * 0.5
    Higher is better. Considers BOTH tasks equally.
    Overwrites previous best - ensures only 1 model exists at any time.
    Also removes any checkpoint-* directories after each evaluation.
    """
    def __init__(self, output_dir, regional_model):
        self.output_dir = output_dir
        self.regional_model = regional_model
        self.best_score = float('-inf')
        self.best_wer = float('inf')
        self.best_dialect_acc = 0.0

    def _compute_combined_score(self, wer, dialect_acc):
        """
        Combined score: weighs both ASR and Classification equally.
        Score = (100 - WER) * 0.5 + Dialect_Accuracy * 0.5
        """
        asr_score = 100 - wer  # Lower WER = higher score
        return asr_score * 0.5 + dialect_acc * 0.5

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        current_wer = metrics.get('eval_wer', float('inf'))
        dialect_acc = metrics.get('eval_dialect_accuracy', 0.0)
        current_score = self._compute_combined_score(current_wer, dialect_acc)

        if current_score > self.best_score:
            self.best_score = current_score
            self.best_wer = current_wer
            self.best_dialect_acc = dialect_acc
            print(f"\n*** New best combined score: {current_score:.2f} ***")
            print(f"    WER: {current_wer:.2f}%, Dialect Acc: {dialect_acc:.2f}%")
            print(f"    Saving model...")

            # Save the regional model (overwrites previous)
            self.regional_model.save_pretrained(
                os.path.join(self.output_dir, "regional_adapter")
            )
        else:
            print(f"\n*** Score {current_score:.2f} (WER: {current_wer:.2f}%, Dialect Acc: {dialect_acc:.2f}%) ***")
            print(f"    Did not improve from best score {self.best_score:.2f}")

        # ALWAYS clean up any checkpoint-* directories after evaluation
        cleanup_checkpoints(self.output_dir)

    def on_save(self, args, state, control, **kwargs):
        """Called when trainer tries to save - immediately clean up."""
        cleanup_checkpoints(self.output_dir)


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

    # Clean up any leftover checkpoints from previous runs
    removed = cleanup_checkpoints(OUTPUT_DIR)
    if removed > 0:
        print(f"Cleaned up {removed} leftover checkpoint(s) from previous run")

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

    # Compute class counts for weighted loss (order: Northern, Southern, Western, Central)
    from collections import Counter
    dialect_counts = Counter(train_dialects)
    class_counts = [
        dialect_counts.get("Northern_Dialect", 0),
        dialect_counts.get("Southern_Dialect", 0),
        dialect_counts.get("Western_Dialect", 0),
        dialect_counts.get("Central_Dialect", 0)
    ]
    print(f"\nClass distribution (train): {dict(dialect_counts)}")
    print(f"Class counts for weighting: {class_counts}")

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
        save_strategy="no",  # NO checkpoints during training - only final model

        logging_steps=LOGGING_STEPS,

        learning_rate=LEARNING_RATE,
        num_train_epochs=NUM_EPOCHS,
        weight_decay=WEIGHT_DECAY,
        warmup_steps=WARMUP_STEPS,

        fp16=FP16 and torch.cuda.is_available(),
        predict_with_generate=True,

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

    # Create callbacks
    best_model_callback = SaveBestModelCallback(OUTPUT_DIR, regional_model)
    epoch_callback = EpochUpdateCallback(train_dataset)  # For curriculum augmentation

    trainer = RegionalTrainer(
        regional_model=regional_model,
        class_counts=class_counts,  # For weighted classification loss
        model=regional_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=processor,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[best_model_callback, epoch_callback],
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
    # Save Training Artifacts (Best model already saved by callback)
    # =========================================================================
    print("\n" + "=" * 80)
    print("Saving Training Artifacts")
    print("=" * 80)

    # Save trainer state (for potential resume)
    trainer.save_state()

    # NOTE: Best regional model already saved by SaveBestModelCallback
    # DO NOT save regional_model here as it would overwrite the best model with the last model
    print(f"Best model (WER: {best_model_callback.best_wer:.2f}%) saved at: {os.path.join(OUTPUT_DIR, 'regional_adapter')}")

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

    print(f"\nTraining artifacts saved to: {OUTPUT_DIR}")

    # =========================================================================
    # Cleanup any checkpoints (use shared function)
    # =========================================================================
    cleanup_checkpoints(OUTPUT_DIR)

    # =========================================================================
    # Final Evaluation
    # =========================================================================
    print("\n" + "=" * 80)
    print("Final Evaluation on Validation Set")
    print("=" * 80)

    eval_results = trainer.evaluate()
    trainer.log_metrics("eval", eval_results)
    trainer.save_metrics("eval", eval_results)

    # Final cleanup after evaluation (in case it triggered any saves)
    cleanup_checkpoints(OUTPUT_DIR)

    final_wer = eval_results.get('eval_wer', 0)
    final_dialect_acc = eval_results.get('eval_dialect_accuracy', 0)
    final_score = (100 - final_wer) * 0.5 + final_dialect_acc * 0.5

    print(f"\nFinal validation (last model):")
    print(f"  - WER: {final_wer:.2f}%")
    print(f"  - Dialect Accuracy: {final_dialect_acc:.2f}%")
    print(f"  - Combined Score: {final_score:.2f}")
    print(f"\nBest validation (saved model):")
    print(f"  - WER: {best_model_callback.best_wer:.2f}%")
    print(f"  - Dialect Accuracy: {best_model_callback.best_dialect_acc:.2f}%")
    print(f"  - Combined Score: {best_model_callback.best_score:.2f}")

    print("\n" + "=" * 80)
    print("Training Complete!")
    print(f"Best model saved at: {os.path.join(OUTPUT_DIR, 'regional_adapter')}")
    print("=" * 80)


if __name__ == "__main__":
    main()
