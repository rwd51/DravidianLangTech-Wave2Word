"""
Model classes for Tamil Dialect Classification using Whisper with Regional Adapter
"""
import torch
import torch.nn as nn
from transformers.modeling_outputs import BaseModelOutput

from config import ADAPTER_DIM, NUM_REGIONS


class RegionalAdapterWhisper(nn.Module):
    """
    Lightweight adapter that adds regional/dialect information to Whisper.
    Adds minimal parameters for computational efficiency.

    This model combines:
    1. Whisper encoder for speech feature extraction
    2. Regional embedding layer for dialect-specific information
    3. Dialect classification head for auxiliary task
    4. Whisper decoder for ASR (speech-to-text)
    """

    def __init__(
        self,
        original_whisper,
        num_regions: int = NUM_REGIONS,
        adapter_dim: int = ADAPTER_DIM
    ):
        """
        Initialize the Regional Adapter Whisper model.

        Args:
            original_whisper: Pre-trained Whisper model
            num_regions: Number of dialect regions/classes
            adapter_dim: Dimension of the regional adapter embedding
        """
        super().__init__()
        self.whisper = original_whisper
        self.num_regions = num_regions
        self.adapter_dim = adapter_dim

        # Small regional adapter - adds minimal parameters
        self.region_embedding = nn.Embedding(num_regions, adapter_dim)

        # Lightweight projection to match Whisper's dimension
        self.adapter_projection = nn.Linear(
            adapter_dim, original_whisper.config.d_model
        )

        # Small classification head for auxiliary dialect classification task
        self.region_classifier = nn.Linear(
            original_whisper.config.d_model, num_regions
        )

        # Layer norm for stability
        self.layer_norm = nn.LayerNorm(original_whisper.config.d_model)

        # Print model info
        num_adapter_params = (
            (adapter_dim * num_regions) +
            (adapter_dim * original_whisper.config.d_model) +
            (original_whisper.config.d_model * num_regions)
        )
        print(f"Regional Adapter initialized:")
        print(f"  - Number of regions: {num_regions}")
        print(f"  - Adapter dimension: {adapter_dim}")
        print(f"  - Additional parameters: ~{num_adapter_params:,}")

    def forward(
        self,
        input_features,
        region_idx=None,
        decoder_input_ids=None,
        labels=None
    ):
        """
        Forward pass through the Regional Adapter Whisper model.

        Args:
            input_features: Audio features from the feature extractor
            region_idx: Regional/dialect indices for adaptation
            decoder_input_ids: Decoder input IDs for teacher forcing
            labels: Target labels for computing loss

        Returns:
            Dictionary containing:
                - loss: Combined ASR loss
                - asr_logits: Logits for ASR predictions
                - region_logits: Logits for dialect classification
                - encoder_hidden_states: Adapted encoder hidden states
        """
        # Get regional embedding if region index is provided
        if region_idx is not None:
            # [batch_size, adapter_dim]
            region_emb = self.region_embedding(region_idx)
            # [batch_size, d_model]
            region_emb = self.adapter_projection(region_emb)
            region_emb = self.layer_norm(region_emb)
        else:
            region_emb = None

        # Forward pass through Whisper encoder
        encoder_outputs = self.whisper.model.encoder(input_features=input_features)
        hidden_states = encoder_outputs.last_hidden_state

        # Add regional information to encoder outputs (simple addition)
        if region_emb is not None:
            # Expand region embedding to match sequence length and add
            region_emb_expanded = region_emb.unsqueeze(1).expand(
                -1, hidden_states.size(1), -1
            )
            adapted_hidden_states = hidden_states + region_emb_expanded
        else:
            adapted_hidden_states = hidden_states

        # Regional classification (auxiliary task)
        region_logits = None
        if region_idx is not None:
            # Use mean pooled representation for classification
            pooled_output = adapted_hidden_states.mean(dim=1)
            region_logits = self.region_classifier(pooled_output)

        # Continue with decoder if needed
        decoder_outputs = None
        loss = None

        if labels is not None or decoder_input_ids is not None:
            # Create a generic encoder output that Whisper decoder can use
            adapted_encoder_outputs = BaseModelOutput(
                last_hidden_state=adapted_hidden_states,
                hidden_states=encoder_outputs.hidden_states
                if hasattr(encoder_outputs, 'hidden_states') else None,
                attentions=encoder_outputs.attentions
                if hasattr(encoder_outputs, 'attentions') else None
            )

            # Forward through decoder
            if labels is not None:
                # Training mode - use labels
                decoder_outputs = self.whisper(
                    encoder_outputs=(adapted_encoder_outputs,),
                    decoder_input_ids=decoder_input_ids,
                    labels=labels,
                    return_dict=True
                )
            else:
                # Inference mode - just decoder_input_ids
                decoder_outputs = self.whisper(
                    encoder_outputs=(adapted_encoder_outputs,),
                    decoder_input_ids=decoder_input_ids,
                    return_dict=True
                )

            loss = decoder_outputs.loss if hasattr(decoder_outputs, 'loss') else None

        return {
            "loss": loss,
            "asr_logits": decoder_outputs.logits
            if decoder_outputs and hasattr(decoder_outputs, 'logits') else None,
            "region_logits": region_logits,
            "encoder_hidden_states": adapted_hidden_states
        }

    def generate(self, input_features, **kwargs):
        """
        Generate transcription using the Whisper model.

        Args:
            input_features: Audio features
            **kwargs: Additional arguments passed to generate

        Returns:
            Generated token IDs
        """
        return self.whisper.generate(input_features, **kwargs)

    def save_pretrained(self, save_path: str):
        """
        Save the model state dict and configuration.

        Args:
            save_path: Path to save the model
        """
        import os
        import json

        os.makedirs(save_path, exist_ok=True)

        # Save the full state dict
        torch.save(self.state_dict(), os.path.join(save_path, "model.pt"))

        # Save configuration
        config = {
            "num_regions": self.num_regions,
            "adapter_dim": self.adapter_dim,
            "whisper_config": self.whisper.config.to_dict()
        }
        with open(os.path.join(save_path, "adapter_config.json"), "w") as f:
            json.dump(config, f, indent=2)

        print(f"Model saved to {save_path}")

    @classmethod
    def from_pretrained(cls, load_path: str, base_whisper):
        """
        Load a saved model.

        Args:
            load_path: Path to load the model from
            base_whisper: Base Whisper model to wrap

        Returns:
            Loaded RegionalAdapterWhisper model
        """
        import os
        import json

        # Load configuration
        with open(os.path.join(load_path, "adapter_config.json"), "r") as f:
            config = json.load(f)

        # Create model
        model = cls(
            original_whisper=base_whisper,
            num_regions=config["num_regions"],
            adapter_dim=config["adapter_dim"]
        )

        # Load state dict
        state_dict = torch.load(
            os.path.join(load_path, "model.pt"),
            map_location="cpu"
        )
        model.load_state_dict(state_dict)

        print(f"Model loaded from {load_path}")
        return model


def create_regional_model(whisper_model, num_regions: int = NUM_REGIONS, adapter_dim: int = ADAPTER_DIM):
    """
    Create a Regional Adapter Whisper model.

    Args:
        whisper_model: Base Whisper model
        num_regions: Number of dialect regions
        adapter_dim: Adapter embedding dimension

    Returns:
        RegionalAdapterWhisper model
    """
    return RegionalAdapterWhisper(
        original_whisper=whisper_model,
        num_regions=num_regions,
        adapter_dim=adapter_dim
    )
