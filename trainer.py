"""
Custom Trainer class for Tamil Dialect Classification with Regional Adapter
"""
import torch
import torch.nn as nn
import numpy as np
from transformers import Seq2SeqTrainer

from config import ALPHA_REGION_LOSS


def compute_class_weights(class_counts, device, smoothing=0.1):
    """
    Compute inverse frequency class weights with smoothing.

    Args:
        class_counts: List of sample counts per class [N, S, W, C]
        device: torch device
        smoothing: Smoothing factor to prevent extreme weights

    Returns:
        Normalized class weights tensor
    """
    total = sum(class_counts)
    # Inverse frequency: classes with fewer samples get higher weight
    weights = [total / (len(class_counts) * count) for count in class_counts]
    # Apply smoothing: weight = (1-s) * computed_weight + s * 1.0
    weights = [(1 - smoothing) * w + smoothing for w in weights]
    # Normalize so mean weight = 1
    mean_weight = sum(weights) / len(weights)
    weights = [w / mean_weight for w in weights]
    return torch.tensor(weights, dtype=torch.float32, device=device)


class RegionalTrainer(Seq2SeqTrainer):
    """
    Custom trainer that handles regional/dialect classification as auxiliary task.
    Combines ASR loss with dialect classification loss.
    Uses class weighting to handle imbalanced dialect distribution.
    """

    def __init__(self, regional_model, class_counts=None, alpha: float = ALPHA_REGION_LOSS, *args, **kwargs):
        """
        Initialize the Regional Trainer.

        Args:
            regional_model: The RegionalAdapterWhisper model
            class_counts: List of sample counts per class for weighting [N, S, W, C]
            alpha: Weight for regional classification loss (0-1)
            *args, **kwargs: Arguments passed to Seq2SeqTrainer
        """
        super().__init__(*args, **kwargs)
        self.regional_model = regional_model
        self.alpha = alpha  # Weight for regional classification loss
        self.class_counts = class_counts

        # Initialize criterion (will be set with weights on first forward pass)
        self._region_criterion = None

    def _remove_unused_columns(self, dataset, description=None):
        """
        Override to keep all columns - we need region_labels and region_idx.
        """
        # Don't remove any columns - we need them all!
        return dataset

    def _get_region_criterion(self, device):
        """Get or create the region criterion with class weights."""
        if self._region_criterion is None:
            if self.class_counts is not None:
                weights = compute_class_weights(self.class_counts, device)
                print(f"Using class weights: {weights.tolist()}")
                self._region_criterion = nn.CrossEntropyLoss(weight=weights)
            else:
                self._region_criterion = nn.CrossEntropyLoss()
        return self._region_criterion

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Compute loss with regional classification auxiliary task.

        Args:
            model: The model
            inputs: Dictionary of inputs
            return_outputs: Whether to return outputs
            num_items_in_batch: Number of items in batch (for newer transformers versions)

        Returns:
            Loss tensor (and optionally outputs)
        """
        # Forward pass with regional information
        outputs = self.regional_model(
            input_features=inputs["input_features"],
            region_idx=inputs["region_idx"],
            decoder_input_ids=inputs.get("decoder_input_ids"),
            labels=inputs["labels"]
        )

        # ASR loss (from Whisper decoder)
        asr_loss = outputs["loss"] if outputs["loss"] is not None else torch.tensor(0.0)

        # Regional classification loss (with class weighting)
        region_loss = torch.tensor(0.0, device=asr_loss.device)
        if outputs["region_logits"] is not None and "region_labels" in inputs:
            criterion = self._get_region_criterion(asr_loss.device)
            region_loss = criterion(
                outputs["region_logits"],
                inputs["region_labels"]
            )

        # Combined loss
        total_loss = asr_loss + self.alpha * region_loss

        return (total_loss, outputs) if return_outputs else total_loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        """
        Override prediction step to handle regional model.
        """
        has_labels = "labels" in inputs
        inputs = self._prepare_inputs(inputs)

        if prediction_loss_only:
            with torch.no_grad():
                outputs = self.regional_model(
                    input_features=inputs["input_features"],
                    region_idx=inputs.get("region_idx"),
                    labels=inputs.get("labels")
                )
                loss = outputs["loss"]
            return (loss, None, None)

        # Generate predictions
        with torch.no_grad():
            generated_tokens = self.regional_model.whisper.generate(
                inputs["input_features"],
                max_length=self.args.generation_max_length or 225
            )

            if has_labels:
                outputs = self.regional_model(
                    input_features=inputs["input_features"],
                    region_idx=inputs.get("region_idx"),
                    labels=inputs["labels"]
                )
                loss = outputs["loss"]
            else:
                loss = None

        # Pad generated tokens if needed
        if generated_tokens.shape[-1] < self.args.generation_max_length:
            generated_tokens = self._pad_tensors_to_max_len(
                generated_tokens, self.args.generation_max_length
            )

        labels = inputs.get("labels")
        if labels is not None and labels.shape[-1] < self.args.generation_max_length:
            labels = self._pad_tensors_to_max_len(
                labels, self.args.generation_max_length
            )

        return (loss, generated_tokens, labels)

    def _pad_tensors_to_max_len(self, tensor, max_length):
        """Pad tensors to max length."""
        if tensor.shape[-1] >= max_length:
            return tensor

        # Use model's pad_token_id or default to eos_token_id
        pad_token_id = self.regional_model.whisper.config.pad_token_id
        if pad_token_id is None:
            pad_token_id = self.regional_model.whisper.config.eos_token_id
        if pad_token_id is None:
            pad_token_id = 0

        padded = torch.full(
            (tensor.shape[0], max_length),
            pad_token_id,
            dtype=tensor.dtype,
            device=tensor.device
        )
        padded[:, :tensor.shape[-1]] = tensor
        return padded

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        """
        Override evaluate to also compute dialect classification accuracy.
        """
        # First, run the standard evaluation (WER)
        metrics = super().evaluate(
            eval_dataset=eval_dataset,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix
        )

        # Now compute classification accuracy on eval dataset
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        if eval_dataset is None:
            return metrics

        self.regional_model.eval()
        all_preds = []
        all_labels = []

        eval_dataloader = self.get_eval_dataloader(eval_dataset)

        with torch.no_grad():
            for batch in eval_dataloader:
                batch = self._prepare_inputs(batch)

                # Forward pass to get classification logits
                outputs = self.regional_model(
                    input_features=batch["input_features"],
                    region_idx=batch.get("region_idx"),
                    labels=batch.get("labels")
                )

                if outputs["region_logits"] is not None and "region_labels" in batch:
                    preds = torch.argmax(outputs["region_logits"], dim=-1)
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(batch["region_labels"].cpu().numpy())

        # Compute classification accuracy
        if len(all_preds) > 0:
            all_preds = np.array(all_preds)
            all_labels = np.array(all_labels)
            accuracy = (all_preds == all_labels).mean() * 100
            metrics[f"{metric_key_prefix}_dialect_accuracy"] = accuracy
            print(f"\n  Dialect Classification Accuracy: {accuracy:.2f}%")

        return metrics


def compute_metrics_factory(processor, metric):
    """
    Create a compute_metrics function for the trainer.

    Args:
        processor: Whisper processor for decoding
        metric: Evaluation metric (e.g., WER)

    Returns:
        compute_metrics function
    """
    def compute_metrics(pred):
        """Compute WER metric."""
        pred_ids = pred.predictions
        label_ids = pred.label_ids

        # Replace -100 with pad token id
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

        # Decode predictions and labels
        pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.batch_decode(label_ids, skip_special_tokens=True)

        # Compute WER
        wer = 100 * metric.compute(predictions=pred_str, references=label_str)

        return {"wer": wer}

    return compute_metrics
