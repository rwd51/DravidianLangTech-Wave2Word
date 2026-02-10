#!/usr/bin/env python3
"""
Fine-tuning script for Dolphin speech recognition model on Tamil dialect data.
"""

import os
import sys
import logging
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
import warnings

warnings.filterwarnings("ignore")

# Setup path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from finetune_config import (
    DATASET_ROOT,
    PRETRAINED_MODEL_DIR,
    OUTPUT_DIR,
    CHECKPOINT_DIR,
    LEARNING_RATES,
    WARMUP_STEPS,
    CTC_WEIGHT,
    DROPOUT_RANGE,
    NUM_EPOCHS,
    BATCH_SIZE,
    GRADIENT_ACCUMULATION_STEPS,
    MAX_GRAD_NORM,
    VALIDATION_SPLIT,
    VAL_CHECK_INTERVAL,
    LOG_INTERVAL,
    DEVICE,
    DTYPE,
    NUM_WORKERS,
    TRANSCRIPTION_OUTPUT,
    SEED,
    format_token_sequence,
    SPECIAL_TOKENS,
)
from finetune_dataclass import AudioDataLoader, AudioSample
from finetune_utils import (
    setup_training_config,
    compute_wer,
    TrainingMetrics,
    save_checkpoint,
    load_checkpoint,
    log_metrics,
    get_device,
)
from dolphin.model import DolphinSpeech2Text
from dolphin.audio import load_audio

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
)
logger = logging.getLogger(__name__)


class AudioDataset(Dataset):
    """PyTorch Dataset for audio samples."""

    def __init__(self, samples, device="cuda"):
        self.samples = samples
        self.device = device

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        return {
            "audio": torch.from_numpy(sample.audio_data).float(),
            "transcription": sample.get_token_sequence(),
            "original_transcription": sample.transcription,
            "filenames": sample.original_filenames,
            "duration": sample.duration,
        }


def collate_batch(batch):
    """Custom collate function for variable-length audio."""
    # Find max length in batch
    max_audio_len = max(item["audio"].shape[0] for item in batch)

    # Pad all audios to max length
    padded_audios = []
    for item in batch:
        audio = item["audio"]
        if audio.shape[0] < max_audio_len:
            padding = torch.zeros(max_audio_len - audio.shape[0])
            audio = torch.cat([audio, padding])
        padded_audios.append(audio)

    return {
        "audios": torch.stack(padded_audios),
        "transcriptions": [item["transcription"] for item in batch],
        "original_transcriptions": [item["original_transcription"] for item in batch],
        "filenames_list": [item["filenames"] for item in batch],
        "durations": [item["duration"] for item in batch],
    }


def load_pretrained_model(model_dir, device):
    """Load pretrained Dolphin model."""
    logger.info("Loading pretrained model...")

    model_dir = Path(model_dir)
    config_path = model_dir / "config.yaml"
    bpe_model_path = model_dir / "bpe.model"
    feats_stats_path = model_dir / "feats_stats.npz"
    model_file = model_dir / "small.pt"

    assert config_path.exists(), f"Config not found: {config_path}"
    assert bpe_model_path.exists(), f"BPE model not found: {bpe_model_path}"
    assert feats_stats_path.exists(), f"Feats stats not found: {feats_stats_path}"
    assert model_file.exists(), f"Model file not found: {model_file}"

    # Import load function
    import yaml
    from espnet2.tasks.s2t import S2TTask
    from espnet2.text.build_tokenizer import build_tokenizer
    from espnet2.text.token_id_converter import TokenIDConverter

    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Fix hardcoded stats path - update to use local feats_stats.npz
    if "normalize_conf" in config and "stats_file" in config["normalize_conf"]:
        config["normalize_conf"]["stats_file"] = str(feats_stats_path)
        logger.info(f"Updated stats_file path to: {feats_stats_path}")

        # Write modified config to a temp file for ESPnet to read
        temp_config_path = model_dir / "config_temp.yaml"
        with open(temp_config_path, "w") as f:
            yaml.dump(config, f)
        config_file_to_use = str(temp_config_path)
    else:
        config_file_to_use = str(config_path)

    # Build model using ESPnet
    logger.info("Building model from config...")
    model, train_args = S2TTask.build_model_from_file(str(config_path), str(model_file))

    model.to(device)
    model.train()

    # Build tokenizer and converter for text processing
    logger.info("Loading tokenizer and converter...")
    tokenizer = build_tokenizer(token_type="bpe", bpemodel=str(bpe_model_path))
    converter = TokenIDConverter(token_list=model.token_list)
    
    logger.info(f"Model loaded successfully")
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    logger.info(
        f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}"
    )
    logger.info(f"Vocabulary size: {len(model.token_list)}")

    return model, train_args, tokenizer, converter


def compute_ctc_loss(model, audio, transcription_tokens, train_args, device):
    """Compute CTC loss for a batch."""
    # Convert audio to features using preprocessor
    batch_size = audio.shape[0]

    # Get audio length
    audio_length = torch.tensor(
        [audio.shape[1] for _ in range(batch_size)], dtype=torch.long, device=device
    )

    # Feature extraction
    with torch.no_grad():
        feats, feats_length = model.encode(audio, audio_length)

    # Encoder output
    encoder_out, encoder_out_lens, _ = model.encoder(feats, feats_length)

    # CTC loss computation
    ys_hat = model.ctc.argmax(encoder_out)

    # For actual CTC loss, we need token indices
    # This is a simplified version - in practice you'd need proper tokenization
    ctc_loss = model.criterion_ctc(ys_hat.view(-1), transcription_tokens.view(-1))

    return ctc_loss


def eval_step(model, val_dataloader, device, train_args, tokenizer, converter):
    """Validation step."""
    logger.info("Running validation...")
    logger.info(f"Processing {len(val_dataloader)} validation batches...")

    model.eval()
    total_loss = 0.0
    total_wer = 0.0
    num_batches = 0
    num_samples = 0
    transcriptions = []

    with torch.no_grad():
        pbar = tqdm(val_dataloader, desc="Validation", leave=False)
        for batch in pbar:
            audios = batch["audios"].to(device)
            original_transcriptions = batch["original_transcriptions"]

            # Forward pass (simplified - actual implementation needs proper tokenization)
            try:
                # Get encoder output
                audio_length = torch.tensor(
                    [audios.shape[1] for _ in range(audios.shape[0])],
                    dtype=torch.long,
                    device=device,
                )

                feats, feats_length = model.encode(audios, audio_length)
                encoder_out, encoder_out_lens, _ = model.encoder(feats, feats_length)

                # Compute CTC loss on validation batch
                blank_id = int(getattr(model.ctc, "blank_id", 0))
                target_tokens_list = []
                target_lengths = []
                for trans_text in original_transcriptions:
                    tokens = tokenizer.text2tokens(trans_text)
                    token_ids = converter.tokens2ids(tokens)
                    if len(token_ids) == 0:
                        raise ValueError("Empty tokenized transcription in validation batch")
                    target_tokens_list.append(
                        torch.tensor(token_ids, dtype=torch.long, device=device)
                    )
                    target_lengths.append(len(token_ids))

                max_target_len = max(target_lengths)
                targets = torch.zeros(
                    len(target_tokens_list),
                    max_target_len,
                    dtype=torch.long,
                    device=device,
                )
                for i, tokens in enumerate(target_tokens_list):
                    targets[i, : len(tokens)] = tokens

                target_lengths = torch.tensor(
                    target_lengths, dtype=torch.long, device=device
                )

                ctc_logits = model.ctc.ctc_lo(encoder_out)
                ctc_log_probs = F.log_softmax(ctc_logits, dim=-1)
                log_probs_ctc = ctc_log_probs.transpose(0, 1)

                batch_loss = F.ctc_loss(
                    log_probs_ctc,
                    targets,
                    encoder_out_lens,
                    target_lengths,
                    blank=blank_id,
                    reduction="mean",
                    zero_infinity=True,
                )
                total_loss += float(batch_loss.item())

                # CTC decoding - get predictions for each sample in batch
                ctc_probs = model.ctc.log_softmax(encoder_out)
                
                for i, (filenames, orig_trans) in enumerate(
                    zip(batch["filenames_list"], original_transcriptions)
                ):
                    # Get CTC predictions for this sample
                    ys_hat = ctc_probs[i].argmax(dim=-1)  # [T]
                    
                    # Remove duplicates and blanks (blank token is typically 0)
                    pred_tokens = []
                    prev_token = -1
                    for token in ys_hat.cpu().tolist():
                        if token != blank_id and token != prev_token:
                            pred_tokens.append(token)
                        prev_token = token
                    
                    # Convert token IDs to text
                    try:
                        # Convert IDs to tokens
                        tokens = converter.ids2tokens(pred_tokens)
                        # Join tokens and detokenize
                        pred_text = tokenizer.tokens2text(tokens)
                    except Exception as e:
                        pred_text = f"[Decoding error: {e}]"
                        logger.debug(f"Decoding error: {e}")
                    
                    transcriptions.append(
                        {
                            "filenames": filenames,
                            "original": orig_trans,
                            "predicted": pred_text,
                        }
                    )
                    num_samples += 1

                num_batches += 1
                pbar.update(1)

            except Exception as e:
                logger.warning(f"Error during validation batch: {e}")
                continue

    pbar.close()
    avg_loss = total_loss / max(num_batches, 1)
    logger.info(f"Validation completed: {num_batches} batches processed, {len(transcriptions)} transcriptions generated")
    logger.info(f"Validation Loss: {avg_loss:.6f}")
    logger.info(f"Sample predictions (first 3):")
    for i, trans in enumerate(transcriptions[:3]):
        logger.info(f"  Sample {i+1}:")
        logger.info(f"    Original: {trans['original'][:100]}...")
        logger.info(f"    Predicted: {trans['predicted'][:100]}...")
    model.train()
    return avg_loss, transcriptions


def train_epoch(
    model, train_dataloader, optimizer, scheduler, device, train_args, epoch, metrics, tokenizer, converter
):
    """Train for one epoch."""
    logger.info(f"Training epoch {epoch + 1}/{NUM_EPOCHS}")

    model.train()
    total_loss = 0.0
    accum_loss = 0.0
    num_batches = 0

    pbar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}", leave=True)

    for batch_idx, batch in enumerate(pbar):
        audios = batch["audios"].to(device)
        original_transcriptions = batch["original_transcriptions"]

        try:
            # Get audio lengths
            audio_length = torch.tensor(
                [audios.shape[1] for _ in range(audios.shape[0])],
                dtype=torch.long,
                device=device,
            )

            # Forward pass through encoder
            feats, feats_length = model.encode(audios, audio_length)

            # Encoder forward
            encoder_out, encoder_out_lens, _ = model.encoder(feats, feats_length)

            # Prepare target tokens for CTC loss
            target_tokens_list = []
            target_lengths = []
            for trans_text in original_transcriptions:
                # Tokenize the transcription text
                tokens = tokenizer.text2tokens(trans_text)
                token_ids = converter.tokens2ids(tokens)
                if len(token_ids) == 0:
                    raise ValueError("Empty tokenized transcription in training batch")
                target_tokens_list.append(
                    torch.tensor(token_ids, dtype=torch.long, device=device)
                )
                target_lengths.append(len(token_ids))
            
            # Pad targets to same length
            max_target_len = max(target_lengths)
            targets = torch.zeros(len(target_tokens_list), max_target_len, dtype=torch.long, device=device)
            for i, tokens in enumerate(target_tokens_list):
                targets[i, :len(tokens)] = tokens
            
            target_lengths = torch.tensor(target_lengths, dtype=torch.long, device=device)
            
            # Compute CTC loss
            # encoder_out shape: [B, T, vocab_size]
            # targets shape: [B, S] where S is target length
            blank_id = int(getattr(model.ctc, "blank_id", 0))
            ctc_logits = model.ctc.ctc_lo(encoder_out)  # Get CTC logits
            ctc_log_probs = F.log_softmax(ctc_logits, dim=-1)  # [B, T, vocab_size]
            
            # CTC loss expects: log_probs (T, B, C), targets (B, S), input_lengths (B), target_lengths (B)
            log_probs_ctc = ctc_log_probs.transpose(0, 1)  # [T, B, vocab_size]
            
            loss = F.ctc_loss(
                log_probs_ctc,
                targets,
                encoder_out_lens,
                target_lengths,
                blank=blank_id,
                reduction="mean",
                zero_infinity=True
            )

            # Normalize loss by gradient accumulation steps
            loss = loss / GRADIENT_ACCUMULATION_STEPS
            loss.backward()

            accum_loss += loss.item()
            total_loss += loss.item()
            num_batches += 1

            # Gradient accumulation
            if (batch_idx + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()

                # Update metrics
                metrics.update_train(accum_loss * GRADIENT_ACCUMULATION_STEPS)
                accum_loss = 0.0

                # Log metrics
                current_lr = optimizer.param_groups[0]["lr"]
                if metrics.step % LOG_INTERVAL == 0:
                    train_loss = metrics.get_train_loss()
                    log_metrics(metrics.step, train_loss, lr=current_lr)

                pbar.set_postfix(
                    {"loss": f"{train_loss:.4f}", "lr": f"{current_lr:.2e}"}
                )

        except Exception as e:
            logger.warning(f"Error processing batch {batch_idx}: {e}")
            optimizer.zero_grad()
            continue

    pbar.close()
    avg_loss = total_loss / max(num_batches, 1)
    logger.info(f"Epoch {epoch + 1} - Average Training Loss: {avg_loss:.6f} (computed from {num_batches} batches)")
    return avg_loss


def main():
    """Main training function."""
    logger.info("=" * 80)
    logger.info("DOLPHIN FINE-TUNING SCRIPT")
    logger.info("=" * 80)

    # Set seed
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # Setup device
    device = get_device(DEVICE)

    # Load dataset
    logger.info("Loading dataset...")
    dataset_loader = AudioDataLoader(
        dataset_root=DATASET_ROOT, combine_short_audios=True
    )
    train_samples, val_samples = dataset_loader.load()

    logger.info(f"Train samples: {len(train_samples)}")
    logger.info(f"Val samples: {len(val_samples)}")

    # Create data loaders
    train_dataset = AudioDataset(train_samples, device=device)
    val_dataset = AudioDataset(val_samples, device=device)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_batch,
        num_workers=0,  # Set to 0 for audio data
        pin_memory=True,
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_batch,
        num_workers=0,
        pin_memory=True,
    )

    # Load pretrained model
    model, train_args, tokenizer, converter = load_pretrained_model(PRETRAINED_MODEL_DIR, device)

    # Setup training
    optimizer, scheduler, total_steps, training_config = setup_training_config(
        model, train_dataloader, NUM_EPOCHS, WARMUP_STEPS, LEARNING_RATES
    )

    metrics = TrainingMetrics(log_interval=LOG_INTERVAL)

    logger.info("Training configuration:")
    for key, value in training_config.items():
        logger.info(f"  {key}: {value}")

    # Training loop
    logger.info("Starting training...")
    best_wer = float("inf")

    for epoch in range(NUM_EPOCHS):
        # Train
        train_loss = train_epoch(
            model,
            train_dataloader,
            optimizer,
            scheduler,
            device,
            train_args,
            epoch,
            metrics,
            tokenizer,
            converter,
        )

        # Validation (every VAL_CHECK_INTERVAL steps or at end of epoch)
        if (epoch + 1) % 1 == 0:  # Validate every epoch
            val_loss, transcriptions = eval_step(
                model, val_dataloader, device, train_args, tokenizer, converter
            )
            metrics.add_val_metrics(val_loss, 0.0)  # Placeholder WER

            val_loss_val, wer_val = metrics.get_last_val_metrics()
            log_metrics(metrics.step, train_loss, val_loss=val_loss_val, wer=wer_val)

            # Print epoch summary
            print(f"\n{'='*80}")
            print(f"EPOCH {epoch + 1}/{NUM_EPOCHS} SUMMARY")
            print(f"{'='*80}")
            print(f"Train Loss: {train_loss:.6f}")
            print(f"Val Loss:   {val_loss_val:.6f}")
            print(f"WER:        {wer_val:.4f} (placeholder)")
            print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.2e}")
            print(f"{'='*80}\n")

            # Save checkpoint
            save_checkpoint(
                model,
                optimizer,
                scheduler,
                training_config,
                metrics,
                CHECKPOINT_DIR,
                epoch,
            )

        logger.info(f"Epoch {epoch + 1}/{NUM_EPOCHS} completed")

    logger.info("=" * 80)
    logger.info("All training epochs completed")
    logger.info("=" * 80)

    # Save final model
    logger.info("Saving final model...")
    output_path = Path(OUTPUT_DIR)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save model state
    torch.save(model.state_dict(), output_path / "model.pt")
    logger.info(f"Model weights saved to {output_path / 'model.pt'}")

    # Copy config and assets
    import shutil

    for file in ["config.yaml", "bpe.model", "feats_stats.npz"]:
        src = Path(PRETRAINED_MODEL_DIR) / file
        if src.exists():
            shutil.copy(src, output_path / file)
            logger.info(f"Copied {file} to {output_path}")

    logger.info(f"Model saved to {output_path}")

    # Run final inference on validation set
    logger.info("Running final inference on validation set...")
    val_loss, transcriptions = eval_step(model, val_dataloader, device, train_args, tokenizer, converter)

    # Write transcriptions to file
    logger.info(
        f"Writing {len(transcriptions)} transcriptions to {TRANSCRIPTION_OUTPUT}"
    )
    with open(TRANSCRIPTION_OUTPUT, "w", encoding="utf-8") as f:
        f.write("filename\toriginal_transcription\tmodel_transcription\n")
        f.write("-" * 100 + "\n")
        for trans in transcriptions:
            filenames_str = (
                ",".join(trans["filenames"])
                if isinstance(trans["filenames"], list)
                else trans["filenames"]
            )
            f.write(f"{filenames_str}\t{trans['original']}\t{trans['predicted']}\n")
    logger.info(f"Transcriptions written to {TRANSCRIPTION_OUTPUT}")

    logger.info("=" * 80)
    logger.info("TRAINING COMPLETED")
    logger.info("=" * 80)
    logger.info(f"Output directory: {OUTPUT_DIR}")
    logger.info(f"Checkpoint directory: {CHECKPOINT_DIR}")
    logger.info(f"Transcriptions: {TRANSCRIPTION_OUTPUT}")


if __name__ == "__main__":
    main()
