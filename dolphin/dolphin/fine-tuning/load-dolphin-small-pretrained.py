"""
Load Dolphin pretrained model with external assets and run inference on audio.

This script demonstrates how to:
1. Load the pretrained model from external files (bpe.model, feats_stats.npz, small.pt, config.yaml)
2. Run inference on a test audio file
3. Transcribe speech in Bengali (bn) language

CLI EXAMPLES:
=============

# Basic usage with Bengali language (default)
python dolphin/load-dolphin-small-pretrained.py --audio /path/to/audio.wav

# Specify custom paths for model assets
python dolphin/load-dolphin-small-pretrained.py \
    --audio /path/to/audio.wav \
    --model-dir /path/to/pretrained/assets

# Use a different language (e.g., Hindi)
python dolphin/load-dolphin-small-pretrained.py \
    --audio /path/to/audio.wav \
    --language hi

# Save output to file
python dolphin/load-dolphin-small-pretrained.py \
    --audio /path/to/audio.wav \
    --output /path/to/output.txt

# Verbose output for debugging
python dolphin/load-dolphin-small-pretrained.py \
    --audio /path/to/audio.wav \
    --verbose
"""

import os
import sys
import yaml
import torch
import logging
import argparse
from pathlib import Path
from typing import Union, Optional

# Add parent directory to path so we can import dolphin
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
)
logger = logging.getLogger("dolphin")

from dolphin.model import DolphinSpeech2Text
from dolphin.audio import load_audio


def load_external_assets(
    bpe_model_path: Union[str, Path],
    feats_stats_path: Union[str, Path],
    config_path: Union[str, Path],
) -> tuple:
    """
    Load external asset files.
    
    Args:
        bpe_model_path: Path to bpe.model file
        feats_stats_path: Path to feats_stats.npz file
        config_path: Path to config.yaml file
        
    Returns:
        Tuple of (bpe_model_path, feats_stats_path, config_dict)
    """
    logger.info(f"Loading BPE model from: {bpe_model_path}")
    assert Path(bpe_model_path).exists(), f"BPE model not found at {bpe_model_path}"
    
    logger.info(f"Loading feature stats from: {feats_stats_path}")
    assert Path(feats_stats_path).exists(), f"Feature stats not found at {feats_stats_path}"
    
    logger.info(f"Loading config from: {config_path}")
    assert Path(config_path).exists(), f"Config file not found at {config_path}"
    
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    return str(bpe_model_path), str(feats_stats_path), config


def create_model_config_from_yaml(config_dict: dict) -> dict:
    """
    Create model configuration dictionary from yaml config.
    
    Args:
        config_dict: Configuration dictionary from config.yaml
        
    Returns:
        Model config dict with encoder and decoder configuration
    """
    # Extract encoder and decoder configurations from yaml
    model_config = {
        "encoder": {
            "output_size": config_dict.get("encoder_conf", {}).get("output_size", 768),
            "attention_heads": config_dict.get("encoder_conf", {}).get("attention_heads", 12),
            "cgmlp_linear_units": config_dict.get("encoder_conf", {}).get("cgmlp_linear_units", 3072),
            "num_blocks": config_dict.get("encoder_conf", {}).get("num_blocks", 12),
            "linear_units": config_dict.get("encoder_conf", {}).get("linear_units", 1536),
        },
        "decoder": {
            "attention_heads": config_dict.get("decoder_conf", {}).get("attention_heads", 12),
            "linear_units": config_dict.get("decoder_conf", {}).get("linear_units", 3072),
            "num_blocks": config_dict.get("decoder_conf", {}).get("num_blocks", 12),
        }
    }
    return model_config


def load_dolphin_model(
    model_dir: Union[str, Path],
    config_path: Union[str, Path],
    bpe_model_path: Union[str, Path],
    feats_stats_path: Union[str, Path],
    device: Optional[str] = None,
    beam_size: int = 5,
) -> DolphinSpeech2Text:
    """
    Load Dolphin model with external assets.
    
    Args:
        model_dir: Directory containing model files
        config_path: Path to config.yaml
        bpe_model_path: Path to bpe.model
        feats_stats_path: Path to feats_stats.npz
        device: Device to use ('cuda', 'cpu', etc.)
        beam_size: Beam size for decoding
        
    Returns:
        Loaded DolphinSpeech2Text model
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    # Load external assets
    bpe_model_path, feats_stats_path, config_dict = load_external_assets(
        bpe_model_path, feats_stats_path, config_path
    )
    
    # Create model configuration
    model_config = create_model_config_from_yaml(config_dict)
    
    # Update config with external asset paths
    config_dict["normalize_conf"]["stats_file"] = feats_stats_path
    
    # Update encoder and decoder configs
    config_dict["encoder_conf"].update(**model_config["encoder"])
    config_dict["decoder_conf"].update(**model_config["decoder"])
    
    # Path to model checkpoint
    model_file = Path(model_dir) / "small.pt"
    assert model_file.exists(), f"Model file not found at {model_file}"
    logger.info(f"Loading model weights from: {model_file}")
    
    # Initialize DolphinSpeech2Text model
    logger.info("Initializing DolphinSpeech2Text model...")
    model = DolphinSpeech2Text(
        s2t_train_config=config_dict,
        s2t_model_file=str(model_file),
        device=device,
        dtype="float32",
        beam_size=beam_size,
        nbest=1,
        task_sym="<asr>",
        predict_time=True,
    )
    
    logger.info("Model loaded successfully!")
    return model


def transcribe_audio(
    model: DolphinSpeech2Text,
    audio_path: Union[str, Path],
    lang_code: str = "bn",
    region_code: Optional[str] = None,
) -> dict:
    """
    Transcribe audio file using the Dolphin model.
    
    Args:
        model: Loaded DolphinSpeech2Text model
        audio_path: Path to audio file
        lang_code: Language code (e.g., 'bn' for Bengali)
        region_code: Region code (optional)
        
    Returns:
        Dictionary with transcription results
    """
    logger.info(f"Loading audio from: {audio_path}")
    audio_path = Path(audio_path)
    assert audio_path.exists(), f"Audio file not found at {audio_path}"
    
    # Load audio
    speech = load_audio(str(audio_path))
    from dolphin.constants import SAMPLE_RATE
    sr = SAMPLE_RATE
    logger.info(f"Audio loaded. Sample rate: {sr}, Duration: {len(speech) / sr:.2f}s")
    
    # Run inference
    logger.info(f"Transcribing with language code: {lang_code}, region: {region_code}")
    result = model(
        speech=speech,
        lang_sym=lang_code.lower(),
        region_sym=region_code,
        padding_speech=False,
    )
    
    return {
        "text": result.text,
        "text_nospecial": result.text_nospecial,
        "language": result.language,
        "region": result.region,
        "rtf": result.rtf,
    }


def main():
    """Main function to load model and run inference."""
    parser = argparse.ArgumentParser(
        description="Load Dolphin pretrained model and run inference"
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default="/kaggle/hf-dolphin-small-pretrained",
        help="Directory containing model files (bpe.model, feats_stats.npz, small.pt, config.yaml)",
    )
    parser.add_argument(
        "--audio",
        type=str,
        required=False,
        help="Path to audio file for transcription",
    )
    parser.add_argument(
        "--lang_code",
        type=str,
        default="bn",
        help="Language code (default: bn for Bengali)",
    )
    parser.add_argument(
        "--region_code",
        type=str,
        default=None,
        help="Region code (optional)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda, cpu, etc.)",
    )
    parser.add_argument(
        "--beam_size",
        type=int,
        default=5,
        help="Beam size for decoding",
    )
    
    args = parser.parse_args()
    
    model_dir = Path(args.model_dir)
    assert model_dir.exists(), f"Model directory not found at {model_dir}"
    
    config_path = model_dir / "config.yaml"
    bpe_model_path = model_dir / "bpe.model"
    feats_stats_path = model_dir / "feats_stats.npz"
    
    # Load model
    logger.info("=" * 60)
    logger.info("Loading Dolphin model with external assets...")
    logger.info("=" * 60)
    
    model = load_dolphin_model(
        model_dir=model_dir,
        config_path=config_path,
        bpe_model_path=bpe_model_path,
        feats_stats_path=feats_stats_path,
        device=args.device,
        beam_size=args.beam_size,
    )
    
    # If audio file is provided, run transcription
    if args.audio:
        logger.info("=" * 60)
        logger.info("Running inference...")
        logger.info("=" * 60)
        
        result = transcribe_audio(
            model=model,
            audio_path=args.audio,
            lang_code=args.lang_code,
            region_code=args.region_code,
        )
        
        logger.info("=" * 60)
        logger.info("Transcription Results:")
        logger.info("=" * 60)
        print(f"\nTranscribed Text:\n{result['text']}\n")
        print(f"Text (no special tokens):\n{result['text_nospecial']}\n")
        print(f"Language: {result['language']}")
        print(f"Region: {result['region']}")
        print(f"RTF (Real Time Factor): {result['rtf']}\n")
    else:
        logger.info("Model loaded successfully. Provide --audio argument to run transcription.")
        logger.info("Example: python load-dolphin-small-pretrained.py --audio /path/to/audio.wav --lang_code bn")


if __name__ == "__main__":
    main()
