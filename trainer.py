"""
Custom Trainer class for Tamil Dialect Classification with Regional Adapter
"""
import torch
import torch.nn as nn
from transformers import Seq2SeqTrainer

from config import ALPHA_REGION_LOSS


class RegionalTrainer(Seq2SeqTrainer):
    """
    Custom trainer that handles regional/dialect classification as auxiliary task.
    Combines ASR loss with dialect classification loss.
    """

    def __init__(self, regional_model, alpha: float = ALPHA_REGION_LOSS, *args, **kwargs):
        """
        Initialize the Regional Trainer.

        Args:
            regional_model: The RegionalAdapterWhisper model
            alpha: Weight for regional classification loss (0-1)
            *args, **kwargs: Arguments passed to Seq2SeqTrainer
        """
        super().__init__(*args, **kwargs)
        self.regional_model = regional_model
        self.region_criterion = nn.CrossEntropyLoss()
        self.alpha = alpha  # Weight for regional classification loss

    def _remove_unused_columns(self, dataset, description=None):
        """
        Override to keep all columns - we need region_labels and region_idx.
        """
        # Don't remove any columns - we need them all!
        return dataset

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

        # Regional classification loss
        region_loss = torch.tensor(0.0, device=asr_loss.device)
        if outputs["region_logits"] is not None and "region_labels" in inputs:
            region_loss = self.region_criterion(
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
