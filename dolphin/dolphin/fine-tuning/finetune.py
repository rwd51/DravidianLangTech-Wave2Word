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

from collections import defaultdict

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
from tamil_text_normalizer import create_normalizer

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
        
        # Debug: verify samples are unique
        if idx < 8 and len(self.samples) > 8:
            # Only log first time during validation (when we have 865 samples)
            if len(self.samples) == 865 and not hasattr(self, '_debug_logged'):
                if idx == 0:
                    self._debug_logged = True
                    print(f"\n[DEBUG][dataset] First 8 validation sample transcriptions:")
                    for i in range(min(8, len(self.samples))):
                        s = self.samples[i]
                        print(f"  Sample {i}: duration={s.duration:.1f}s, trans_len={len(s.transcription)}, text[:60]='{s.transcription[:60]}'")
        
        return {
            "audio": torch.from_numpy(sample.audio_data).float(),
            "transcription": sample.get_token_sequence(),
            "original_transcription": sample.transcription,
            "filenames": sample.original_filenames,
            "duration": sample.duration,
        }


def collate_batch(batch):
    """Custom collate function for variable-length audio."""
    # Debug: Check if all items in batch are identical
    if not hasattr(collate_batch, '_debug_logged'):
        collate_batch._debug_logged = True
        print(f"\n[DEBUG][collate] First batch:")
        print(f"  Batch size: {len(batch)}")
        print(f"  Item types: {type(batch[0])}")
        
        # Check audio uniqueness
        audio_shapes = [item["audio"].shape[0] for item in batch]
        print(f"  Audio shapes: {audio_shapes}")
        
        # Check transcription uniqueness
        trans_lens = [len(item["original_transcription"]) for item in batch]
        print(f"  Transcription lengths: {trans_lens}")
        
        unique_trans = len(set(item["original_transcription"] for item in batch))
        print(f"  Unique transcriptions: {unique_trans}/{len(batch)}")
        
        if unique_trans == 1:
            print(f"  ⚠️ WARNING: All batch items have identical transcriptions!")
            print(f"  First 2 transcriptions:")
            for i in range(min(2, len(batch))):
                print(f"    [{i}]: '{batch[i]['original_transcription'][:80]}'")
        
        # Check if audio data is identical (only if same length)
        if len(batch) >= 2 and batch[0]["audio"].shape == batch[1]["audio"].shape:
            audio_diff = torch.abs(batch[0]["audio"] - batch[1]["audio"]).sum().item()
            print(f"  Audio diff between item 0 and 1: {audio_diff}")
    
    # Find max length in batch
    max_audio_len = max(item["audio"].shape[0] for item in batch)

    # Keep original (unpadded) lengths
    audio_lengths = torch.tensor(
        [item["audio"].shape[0] for item in batch], dtype=torch.long
    )

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
        "audio_lengths": audio_lengths,
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


def eval_step(model, val_dataloader, device, train_args, tokenizer, converter):
    """Validation step."""
    print("\n" + "="*80)
    print("STARTING VALIDATION")
    print("="*80)
    logger.info("Running validation...")
    logger.info(f"Processing {len(val_dataloader)} validation batches...")

    model.eval()
    total_loss = 0.0
    total_wer = 0.0
    num_batches = 0
    num_samples = 0
    num_loss_batches = 0
    num_skipped_loss_batches = 0
    transcriptions = []
    
    # File-level debugging
    file_errors = defaultdict(list)  # filename -> list of error messages
    file_success = set()  # filenames that succeeded
    
    # Prediction statistics for debugging
    pred_stats = {
        "empty": 0,
        "short": 0,
        "decoding_error": 0,
        "valid": 0,
    }
    
    # Store first batch predictions for comparison
    first_batch_pred_tokens = []

    with torch.no_grad():
        pbar = tqdm(val_dataloader, desc="Validation", leave=False)
        for batch_idx, batch in enumerate(pbar):
            audios = batch["audios"].to(device)
            audio_lengths = batch.get("audio_lengths")
            if audio_lengths is not None:
                audio_lengths = audio_lengths.to(device)
            original_transcriptions = batch["original_transcriptions"]

            # Forward pass (simplified - actual implementation needs proper tokenization)
            try:
                # Get encoder output
                if audio_lengths is None:
                    audio_length = torch.tensor(
                        [audios.shape[1] for _ in range(audios.shape[0])],
                        dtype=torch.long,
                        device=device,
                    )
                else:
                    audio_length = audio_lengths

                # Forward pass through encoder (includes frontend, normalize, and encoder)
                encoder_out, encoder_out_lens = model.encode(audios, audio_length)

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

                input_lengths = encoder_out_lens.detach().cpu().tolist()
                target_lengths_list = target_lengths.detach().cpu().tolist()
                valid_indices = [
                    i
                    for i, (tlen, ilen) in enumerate(
                        zip(target_lengths_list, input_lengths)
                    )
                    if tlen > 0 and tlen <= ilen
                ]

                if valid_indices:
                    targets = targets[valid_indices]
                    target_lengths = target_lengths[valid_indices]
                    log_probs_ctc = log_probs_ctc[:, valid_indices, :]
                    input_lengths = torch.tensor(
                        [input_lengths[i] for i in valid_indices],
                        dtype=torch.long,
                        device=device,
                    )

                    batch_loss = F.ctc_loss(
                        log_probs_ctc,
                        targets,
                        input_lengths,
                        target_lengths,
                        blank=blank_id,
                        reduction="mean",
                        zero_infinity=True,
                    )
                    total_loss += float(batch_loss.item())
                    num_loss_batches += 1
                else:
                    num_skipped_loss_batches += 1

                if num_batches == 0:
                    # Debug only for the first validation batch
                    logger.info(
                        "[DEBUG][val] input_len(min/avg/max)=%s / %s / %s, target_len(min/avg/max)=%s / %s / %s, valid=%d/%d",
                        min(input_lengths),
                        int(sum(input_lengths) / max(len(input_lengths), 1)),
                        max(input_lengths),
                        min(target_lengths_list),
                        int(sum(target_lengths_list) / max(len(target_lengths_list), 1)),
                        max(target_lengths_list),
                        len(valid_indices),
                        len(target_lengths_list),
                    )
                    try:
                        sample_text = original_transcriptions[0]
                        sample_tokens = tokenizer.text2tokens(sample_text)
                        logger.info(
                            "[DEBUG][val] sample text (first 80 chars): %s",
                            sample_text[:80],
                        )
                        logger.info(
                            "[DEBUG][val] sample token count=%d, first 10 tokens=%s",
                            len(sample_tokens),
                            sample_tokens[:10],
                        )
                    except Exception as e:
                        logger.info("[DEBUG][val] tokenization debug failed: %s", e)

                # CTC decoding - get predictions for each sample in batch
                ctc_probs = model.ctc.log_softmax(encoder_out)
                
                # Debug first batch CTC outputs
                if batch_idx == 0:
                    print(f"\n[DEBUG][val] ctc_probs shape: {ctc_probs.shape}")
                    print(f"[DEBUG][val] blank_id: {blank_id}")
                    print(f"[DEBUG][val] Batch size: {len(original_transcriptions)}")
                    print(f"[DEBUG][val] Checking if all originals are identical...")
                    all_same = all(t == original_transcriptions[0] for t in original_transcriptions)
                    print(f"[DEBUG][val] All originals identical? {all_same}")
                    if not all_same:
                        print(f"[DEBUG][val] First 3 original transcription lengths: {[len(t) for t in original_transcriptions[:3]]}")
                        print(f"[DEBUG][val] Original 0 (first 100): '{original_transcriptions[0][:100]}'")
                        print(f"[DEBUG][val] Original 1 (first 100): '{original_transcriptions[1][:100]}'")
                        print(f"[DEBUG][val] Original 2 (first 100): '{original_transcriptions[2][:100]}'")
                    else:
                        print(f"[DEBUG][val] ⚠️ WARNING: ALL ORIGINALS ARE IDENTICAL!")
                    
                    # Check if audio tensors are different
                    print(f"[DEBUG][val] Audio tensor shape: {audios.shape}")
                    if audios.shape[0] >= 2:
                        diff = torch.abs(audios[0] - audios[1]).sum().item()
                        print(f"[DEBUG][val] Audio diff between sample 0 and 1: {diff}")
                        if diff < 0.001:
                            print(f"[DEBUG][val] ⚠️ WARNING: AUDIO SAMPLES 0 AND 1 ARE NEARLY IDENTICAL!")
                    
                    # Check if encoder outputs are different
                    print(f"[DEBUG][val] encoder_out shape: {encoder_out.shape}")
                    if encoder_out.shape[0] >= 2:
                        enc_diff = torch.abs(encoder_out[0] - encoder_out[1]).sum().item()
                        print(f"[DEBUG][val] Encoder output diff between sample 0 and 1: {enc_diff}")
                        if enc_diff < 0.001:
                            print(f"[DEBUG][val] ⚠️ WARNING: ENCODER OUTPUTS 0 AND 1 ARE NEARLY IDENTICAL!")
                    
                    # Check if CTC probs are different per sample
                    if ctc_probs.shape[0] >= 2:
                        ctc_diff = torch.abs(ctc_probs[0] - ctc_probs[1]).sum().item()
                        print(f"[DEBUG][val] CTC probs diff between sample 0 and 1: {ctc_diff}")
                        if ctc_diff < 0.001:
                            print(f"[DEBUG][val] ⚠️ WARNING: CTC PROBS 0 AND 1 ARE NEARLY IDENTICAL!")
                    
                    logger.info(f"[DEBUG][val] ctc_probs shape: {ctc_probs.shape}")
                    logger.info(f"[DEBUG][val] blank_id: {blank_id}")
                
                for i, (filenames, orig_trans) in enumerate(
                    zip(batch["filenames_list"], original_transcriptions)
                ):
                    # Get CTC predictions for this sample
                    ys_hat = ctc_probs[i].argmax(dim=-1)  # [T]
                    
                    # Debug first few predictions
                    if batch_idx == 0 and i < 2:
                        print(f"[DEBUG][val] Sample {i} ys_hat shape: {ys_hat.shape}")
                        print(f"[DEBUG][val] Sample {i} ys_hat (first 20): {ys_hat[:20].cpu().tolist()}")
                        print(f"[DEBUG][val] Sample {i} ys_hat (last 20): {ys_hat[-20:].cpu().tolist()}")
                        print(f"[DEBUG][val] Sample {i} unique tokens in ys_hat: {len(set(ys_hat.cpu().tolist()))}")
                        logger.info(f"[DEBUG][val] Sample {i} ys_hat shape: {ys_hat.shape}")
                        logger.info(f"[DEBUG][val] Sample {i} ys_hat (first 20): {ys_hat[:20].cpu().tolist()}")
                    
                    # Remove duplicates and blanks (blank token is typically 0)
                    pred_tokens = []
                    prev_token = -1
                    for token in ys_hat.cpu().tolist():
                        if token != blank_id and token != prev_token:
                            pred_tokens.append(token)
                        prev_token = token
                    
                    # Track prediction length statistics
                    if len(pred_tokens) == 0:
                        pred_stats["empty"] += 1
                    elif len(pred_tokens) < 3:
                        pred_stats["short"] += 1
                    
                    # Store first batch predictions for comparison
                    if batch_idx == 0:
                        first_batch_pred_tokens.append(pred_tokens.copy())
                    
                    # Convert token IDs to text
                    try:
                        # Convert IDs to tokens
                        tokens = converter.ids2tokens(pred_tokens)
                        # Join tokens and detokenize
                        pred_text = tokenizer.tokens2text(tokens)
                        
                        # Debug first few tokenizations
                        if batch_idx == 0 and i < 2:
                            print(f"[DEBUG][val] Sample {i} pred_tokens count: {len(pred_tokens)}")
                            print(f"[DEBUG][val] Sample {i} pred_tokens (first 20): {pred_tokens[:20]}")
                            print(f"[DEBUG][val] Sample {i} tokens (first 10): {tokens[:10]}")
                            print(f"[DEBUG][val] Sample {i} pred_text: '{pred_text}'")
                            print(f"[DEBUG][val] Sample {i} pred_text length: {len(pred_text)}")
                            logger.info(f"[DEBUG][val] Sample {i} pred_tokens count: {len(pred_tokens)}")
                            logger.info(f"[DEBUG][val] Sample {i} tokens (first 10): {tokens[:10]}")
                            logger.info(f"[DEBUG][val] Sample {i} pred_text: '{pred_text}'")
                            logger.info(f"[DEBUG][val] Sample {i} pred_text length: {len(pred_text)}")
                        
                        if pred_text and pred_text.strip():
                            pred_stats["valid"] += 1
                        else:
                            pred_stats["empty"] += 1
                            
                    except Exception as e:
                        pred_text = f"[Decoding error: {e}]"
                        pred_stats["decoding_error"] += 1
                        logger.debug(f"Decoding error: {e}")
                    
                    # Debug comparison for first few samples
                    if batch_idx == 0 and i < 3:
                        print(f"\n[DEBUG][val] === Sample {i} Comparison ===")
                        print(f"  Original: '{orig_trans[:100]}'")
                        print(f"  Predicted: '{pred_text[:100]}'")
                        print(f"  Original length: {len(orig_trans)}, Predicted length: {len(pred_text)}")
                        logger.info(f"[DEBUG][val] === Sample {i} Comparison ===")
                        logger.info(f"  Original: '{orig_trans}'")
                        logger.info(f"  Predicted: '{pred_text}'")
                        logger.info(f"  Original length: {len(orig_trans)}, Predicted length: {len(pred_text)}")
                    
                    transcriptions.append(
                        {
                            "filenames": filenames,
                            "original": orig_trans,
                            "predicted": pred_text,
                        }
                    )
                    num_samples += 1
                    
                    # Track successful files
                    if isinstance(filenames, list):
                        for fname in filenames:
                            file_success.add(fname)
                    else:
                        file_success.add(filenames)

                # After first batch, check if all predictions are identical
                if batch_idx == 0 and len(first_batch_pred_tokens) >= 2:
                    print(f"\n[DEBUG][val] Checking if all predictions in first batch are identical...")
                    all_pred_same = all(
                        first_batch_pred_tokens[0] == pred_tok
                        for pred_tok in first_batch_pred_tokens
                    )
                    print(f"[DEBUG][val] All predictions identical? {all_pred_same}")
                    if all_pred_same:
                        print(f"[DEBUG][val] ⚠️ WARNING: ALL PREDICTIONS IN FIRST BATCH ARE IDENTICAL!")
                        print(f"[DEBUG][val] Number of samples in batch: {len(first_batch_pred_tokens)}")
                    else:
                        print(f"[DEBUG][val] ✓ Predictions are different across samples (good)")

                num_batches += 1
                pbar.update(1)

            except Exception as e:
                logger.warning(f"Error during validation batch: {e}")
                # Track which files were in this failed batch
                for filenames in batch["filenames_list"]:
                    if isinstance(filenames, list):
                        for fname in filenames:
                            file_errors[fname].append(str(e))
                    else:
                        file_errors[filenames].append(str(e))
                continue

    pbar.close()
    
    # Compute average loss from successful batches
    if num_loss_batches > 0:
        avg_loss = total_loss / num_loss_batches
    else:
        avg_loss = 0.0
        logger.warning("No valid validation batches for loss computation!")
    
    logger.info(f"Validation completed: {num_batches} batches processed, {len(transcriptions)} transcriptions generated")
    logger.info(f"Validation Loss: {avg_loss:.6f} (averaged over {num_loss_batches} batches)")
    logger.info(
        f"Validation loss batches used: {num_loss_batches}, skipped (no valid idx): {num_skipped_loss_batches}"
    )
    
    # Log prediction statistics
    print(f"\n" + "="*80)
    print(f"[DEBUG][val] Prediction Statistics:")
    print(f"  Valid predictions: {pred_stats['valid']}")
    print(f"  Empty predictions: {pred_stats['empty']}")
    print(f"  Short predictions (<3 tokens): {pred_stats['short']}")
    print(f"  Decoding errors: {pred_stats['decoding_error']}")
    print(f"  Total: {sum(pred_stats.values())}")
    print("="*80 + "\n")
    logger.info(f"\n[DEBUG][val] Prediction Statistics:")
    logger.info(f"  Valid predictions: {pred_stats['valid']}")
    logger.info(f"  Empty predictions: {pred_stats['empty']}")
    logger.info(f"  Short predictions (<3 tokens): {pred_stats['short']}")
    logger.info(f"  Decoding errors: {pred_stats['decoding_error']}")
    logger.info(f"  Total: {sum(pred_stats.values())}")
    
    # Compute WER
    total_wer = 0.0
    num_wer_samples = 0
    wer_debug_samples = []
    
    for idx, trans in enumerate(transcriptions):
        if trans['predicted'] and not trans['predicted'].startswith('[Decoding error'):
            wer = compute_wer(trans['original'], trans['predicted'])
            total_wer += wer
            num_wer_samples += 1
            
            # Collect debug samples for first 5
            if idx < 5:
                wer_debug_samples.append({
                    "original": trans['original'][:80],
                    "predicted": trans['predicted'][:80],
                    "wer": wer,
                })
    
    avg_wer = total_wer / max(num_wer_samples, 1)
    
    # Log WER computation debug info
    print(f"\n[DEBUG][val] WER Computation Details:")
    for i, sample in enumerate(wer_debug_samples):
        print(f"  Sample {i}: WER={sample['wer']:.4f}")
        print(f"    Original: '{sample['original']}'")
        print(f"    Predicted: '{sample['predicted']}'")
    print(f"\nWER: {avg_wer:.4f} (computed from {num_wer_samples}/{len(transcriptions)} samples)")
    print("="*80 + "\n")
    
    logger.info(f"\n[DEBUG][val] WER Computation Details:")
    for i, sample in enumerate(wer_debug_samples):
        logger.info(f"  Sample {i}: WER={sample['wer']:.4f}")
        logger.info(f"    Original: '{sample['original']}'")
        logger.info(f"    Predicted: '{sample['predicted']}'")
    
    logger.info(f"WER: {avg_wer:.4f} (computed from {num_wer_samples}/{len(transcriptions)} samples)")
    
    logger.info(f"Sample predictions (first 3):")
    for i, trans in enumerate(transcriptions[:3]):
        logger.info(f"  Sample {i+1}:")
        logger.info(f"    Original: {trans['original'][:100]}...")
        logger.info(f"    Predicted: {trans['predicted'][:100]}...")
    
    # Log file-level statistics
    logger.info(f"\n[FILE DEBUG] Validation file statistics:")
    logger.info(f"  Successful files: {len(file_success)}")
    logger.info(f"  Failed files: {len(file_errors)}")
    
    # Write error log if there are failures
    if file_errors:
        error_log_path = Path(OUTPUT_DIR) / "validation_file_errors.txt"
        error_log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(error_log_path, "w", encoding="utf-8") as f:
            f.write(f"Validation File Errors ({len(file_errors)} files)\n")
            f.write("=" * 80 + "\n\n")
            for fname, errors in sorted(file_errors.items()):
                f.write(f"File: {fname}\n")
                f.write(f"Errors: {errors}\n")
                f.write("-" * 80 + "\n")
        logger.info(f"  Error details written to: {error_log_path}")
    
    model.train()
    return avg_loss, transcriptions, avg_wer


def train_epoch(
    model, train_dataloader, optimizer, scheduler, device, train_args, epoch, metrics, tokenizer, converter
):
    """Train for one epoch."""
    logger.info(f"Training epoch {epoch + 1}/{NUM_EPOCHS}")

    model.train()
    total_loss = 0.0
    accum_loss = 0.0
    num_batches = 0
    num_skipped_batches = 0
    
    # File-level debugging
    file_errors = defaultdict(list)  # filename -> list of error messages
    file_success = set()  # filenames that succeeded
    batch_file_map = {}  # batch_idx -> filenames

    pbar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}", leave=True)

    for batch_idx, batch in enumerate(pbar):
        audios = batch["audios"].to(device)
        audio_lengths = batch.get("audio_lengths")
        if audio_lengths is not None:
            audio_lengths = audio_lengths.to(device)
        original_transcriptions = batch["original_transcriptions"]
        
        # Store filenames for this batch (for error tracking)
        batch_file_map[batch_idx] = batch["filenames_list"]

        # DEBUG: Log FIRST, before anything can fail
        if batch_idx == 0:
            logger.info("[DEBUG][train][epoch %d] Starting batch 0, audio shape=%s, batch_size=%d", epoch + 1, audios.shape, len(original_transcriptions))
            logger.info("[DEBUG][train] Sample 0 text preview: %s", original_transcriptions[0][:80] if original_transcriptions else "EMPTY")

        try:
            # Get audio lengths
            if audio_lengths is None:
                audio_length = torch.tensor(
                    [audios.shape[1] for _ in range(audios.shape[0])],
                    dtype=torch.long,
                    device=device,
                )
            else:
                audio_length = audio_lengths

            # Forward pass through encoder (includes frontend, normalize, and encoder)
            encoder_out, encoder_out_lens = model.encode(audios, audio_length)

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
            
            # Compute CTC loss
            # encoder_out shape: [B, T, vocab_size]
            # targets shape: [B, S] where S is target length
            blank_id = int(getattr(model.ctc, "blank_id", 0))
            ctc_logits = model.ctc.ctc_lo(encoder_out)  # Get CTC logits
            ctc_log_probs = F.log_softmax(ctc_logits, dim=-1)  # [B, T, vocab_size]
            
            # CTC loss expects: log_probs (T, B, C), targets (B, S), input_lengths (B), target_lengths (B)
            log_probs_ctc = ctc_log_probs.transpose(0, 1)  # [T, B, vocab_size]

            # Filter out samples where target length exceeds input length
            input_lengths = encoder_out_lens.detach().cpu().tolist()
            target_lengths_list = target_lengths.detach().cpu().tolist()
            
            # DEBUG: Always log first batch stats BEFORE filtering
            if batch_idx == 0:
                logger.info(
                    "[DEBUG][train][epoch %d][BEFORE FILTER] batch_size=%d, input_len(min/avg/max)=%s/%s/%s, target_len(min/avg/max)=%s/%s/%s",
                    epoch + 1,
                    len(input_lengths),
                    min(input_lengths),
                    int(sum(input_lengths) / len(input_lengths)),
                    max(input_lengths),
                    min(target_lengths_list),
                    int(sum(target_lengths_list) / len(target_lengths_list)),
                    max(target_lengths_list),
                )
                # Show actual pairs for first 3 samples
                for i in range(min(3, len(input_lengths))):
                    logger.info(
                        "[DEBUG][train] sample %d: input_len=%d, target_len=%d, text_preview: %s",
                        i,
                        input_lengths[i],
                        target_lengths_list[i],
                        original_transcriptions[i][:60] + "...",
                    )
            
            valid_indices = [
                i
                for i, (tlen, ilen) in enumerate(
                    zip(target_lengths_list, input_lengths)
                )
                if tlen > 0 and tlen <= ilen
            ]

            if not valid_indices:
                if batch_idx < 5:  # Log first 5 skipped batches
                    logger.warning(
                        f"[SKIP] batch {batch_idx}: ALL {len(target_lengths_list)} samples have target_len > input_len"
                    )
                optimizer.zero_grad()
                num_skipped_batches += 1
                continue

            targets = targets[valid_indices]
            target_lengths = target_lengths[valid_indices]
            log_probs_ctc = log_probs_ctc[:, valid_indices, :]
            input_lengths = torch.tensor(
                [input_lengths[i] for i in valid_indices],
                dtype=torch.long,
                device=device,
            )

            loss = F.ctc_loss(
                log_probs_ctc,
                targets,
                input_lengths,
                target_lengths,
                blank=blank_id,
                reduction="mean",
                zero_infinity=True,
            )

            # Normalize loss by gradient accumulation steps
            loss = loss / GRADIENT_ACCUMULATION_STEPS
            loss.backward()

            if batch_idx == 0 or (num_batches == 1):  # Log first successful batch
                # Debug only for the first training batch of the epoch
                logger.info(
                    "[DEBUG][train][epoch %d][AFTER FILTER] loss=%.6f, valid=%d/%d (kept %d%% of batch)",
                    epoch + 1,
                    float(loss.item() * GRADIENT_ACCUMULATION_STEPS),
                    len(valid_indices),
                    len(target_lengths_list),
                    int(100 * len(valid_indices) / len(target_lengths_list)),
                )

            accum_loss += loss.item()
            total_loss += loss.item()
            num_batches += 1
            
            # Track successful files
            for filenames in batch["filenames_list"]:
                if isinstance(filenames, list):
                    for fname in filenames:
                        file_success.add(fname)
                else:
                    file_success.add(filenames)

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
                train_loss = metrics.get_train_loss()  # Always compute for progress bar
                if metrics.step % LOG_INTERVAL == 0:
                    log_metrics(metrics.step, train_loss, lr=current_lr)

                pbar.set_postfix(
                    {"loss": f"{train_loss:.4f}", "lr": f"{current_lr:.2e}"}
                )

        except Exception as e:
            error_msg = f"{type(e).__name__}: {e}"
            if batch_idx < 5:  # Log first 5 errors in detail
                logger.error(f"[ERROR] Batch {batch_idx} failed: {error_msg}")
                import traceback
                logger.error(f"[ERROR] Traceback: {traceback.format_exc()[:500]}")
            elif batch_idx == 5:
                logger.warning(f"(Suppressing further error details, {batch_idx} batches failed so far)")
            
            # Track which files were in this failed batch
            for filenames in batch["filenames_list"]:
                if isinstance(filenames, list):
                    for fname in filenames:
                        file_errors[fname].append(error_msg)
                else:
                    file_errors[filenames].append(error_msg)
            
            optimizer.zero_grad()
            num_skipped_batches += 1
            continue

    pbar.close()
    
    # CRITICAL: Check if ANY batches succeeded
    if num_batches == 0:
        logger.error(f"[CRITICAL] Epoch {epoch + 1}: ZERO successful batches out of {len(train_dataloader)}! All {num_skipped_batches} batches failed.")
        logger.error(f"[CRITICAL] This means either: (1) All exceptions were caught, or (2) All samples filtered out because target_len > input_len")
        return 0.0
    
    avg_loss = total_loss / num_batches
    logger.info(f"Epoch {epoch + 1} - Average Training Loss: {avg_loss:.6f} (computed from {num_batches} successful batches)")
    logger.info(
        f"Epoch {epoch + 1} - Successful: {num_batches}, Skipped/Failed: {num_skipped_batches}, Total: {len(train_dataloader)}"
    )
    
    # Log file-level statistics
    logger.info(f"\n[FILE DEBUG] Training file statistics for Epoch {epoch + 1}:")
    logger.info(f"  Successful files: {len(file_success)}")
    logger.info(f"  Failed files: {len(file_errors)}")
    
    # Write error log if there are failures
    if file_errors:
        error_log_path = Path(OUTPUT_DIR) / f"training_file_errors_epoch{epoch+1}.txt"
        error_log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(error_log_path, "w", encoding="utf-8") as f:
            f.write(f"Training File Errors - Epoch {epoch + 1} ({len(file_errors)} files)\n")
            f.write("=" * 80 + "\n\n")
            for fname, errors in sorted(file_errors.items()):
                f.write(f"File: {fname}\n")
                f.write(f"Errors: {errors}\n")
                f.write("-" * 80 + "\n")
        logger.info(f"  Error details written to: {error_log_path}")
    
    # Write success log
    if file_success:
        success_log_path = Path(OUTPUT_DIR) / f"training_file_success_epoch{epoch+1}.txt"
        with open(success_log_path, "w", encoding="utf-8") as f:
            f.write(f"Successfully Processed Files - Epoch {epoch + 1} ({len(file_success)} files)\n")
            f.write("=" * 80 + "\n\n")
            for fname in sorted(file_success):
                f.write(f"{fname}\n")
        logger.info(f"  Success list written to: {success_log_path}")
    
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
        dataset_root=DATASET_ROOT, combine_short_audios=False
    )
    train_samples, val_samples = dataset_loader.load()

    logger.info(f"Train samples: {len(train_samples)}")
    logger.info(f"Val samples: {len(val_samples)}")
    logger.info("✓ Text normalization applied during data loading (default preset)")

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
        model,
        train_dataloader,
        NUM_EPOCHS,
        WARMUP_STEPS,
        LEARNING_RATES,
        grad_accum_steps=GRADIENT_ACCUMULATION_STEPS,
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
            val_loss, transcriptions, wer = eval_step(
                model, val_dataloader, device, train_args, tokenizer, converter
            )
            metrics.add_val_metrics(val_loss, wer)

            val_loss_val, wer_val = metrics.get_last_val_metrics()
            log_metrics(metrics.step, train_loss, val_loss=val_loss_val, wer=wer_val)

            # Print epoch summary
            print(f"\n{'='*80}")
            print(f"EPOCH {epoch + 1}/{NUM_EPOCHS} SUMMARY")
            print(f"{'='*80}")
            print(f"Train Loss: {train_loss:.6f}")
            print(f"Val Loss:   {val_loss_val:.6f}")
            print(f"WER:        {wer_val:.4f}")
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
    val_loss, transcriptions, wer = eval_step(model, val_dataloader, device, train_args, tokenizer, converter)

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
