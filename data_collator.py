"""
Data Collator classes for Tamil Dialect Classification
Handles dynamic padding for audio features and text labels
"""
from dataclasses import dataclass
from typing import Any, Dict, List, Union
import torch


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    Used for Whisper-based speech-to-text models.
    """
    processor: Any
    decoder_start_token_id: int

    def __call__(
        self,
        features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        """
        Collate a batch of features.

        Args:
            features: List of feature dictionaries from the dataset

        Returns:
            Batched and padded tensors
        """
        # Pad the audio inputs
        input_features = [
            {"input_features": feature["input_features"]}
            for feature in features
        ]
        batch = self.processor.feature_extractor.pad(
            input_features, return_tensors="pt"
        )

        # Get the tokenized label sequences
        label_features = [
            {"input_ids": feature["labels"]}
            for feature in features
        ]

        # Pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(
            label_features, return_tensors="pt"
        )

        # Replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        batch["labels"] = labels
        # Don't manually create decoder_input_ids - let Whisper handle it from labels

        return batch


@dataclass
class DataCollatorRegionalASR(DataCollatorSpeechSeq2SeqWithPadding):
    """
    Enhanced data collator that handles regional/dialect information
    in addition to audio features and text labels.
    """

    def __call__(
        self,
        features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        """
        Collate a batch of features including regional labels.

        Args:
            features: List of feature dictionaries from the dataset

        Returns:
            Batched and padded tensors with regional information
        """
        # IMPORTANT: Extract regional labels BEFORE calling parent
        # because parent method will filter out unexpected keys
        region_labels = [feature["region_labels"] for feature in features]
        region_labels = torch.tensor(region_labels, dtype=torch.long)

        # Also extract region_idx for model input
        region_idx = [feature["region_idx"] for feature in features]
        region_idx = torch.tensor(region_idx, dtype=torch.long)

        # Call parent method for audio and text processing
        batch = super().__call__(features)

        # Add regional information to batch
        batch["region_labels"] = region_labels
        batch["region_idx"] = region_idx

        return batch


def create_data_collator(processor, model_config, include_regional: bool = True):
    """
    Create appropriate data collator.

    Args:
        processor: Whisper processor
        model_config: Model configuration (for decoder_start_token_id)
        include_regional: Whether to include regional labels

    Returns:
        Data collator instance
    """
    decoder_start_token_id = model_config.decoder_start_token_id

    if include_regional:
        return DataCollatorRegionalASR(
            processor=processor,
            decoder_start_token_id=decoder_start_token_id
        )
    else:
        return DataCollatorSpeechSeq2SeqWithPadding(
            processor=processor,
            decoder_start_token_id=decoder_start_token_id
        )
