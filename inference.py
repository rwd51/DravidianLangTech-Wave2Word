"""
Inference Script for Tamil Dialect Classification
Run inference on validation set or test set for analysis
"""
import os
import json
import torch
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
from sklearn.metrics import classification_report, confusion_matrix
import evaluate

from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration
)

from config import (
    MODEL_NAME,
    LANGUAGE,
    TASK,
    OUTPUT_DIR,
    TEST_DIR,
    SAMPLING_RATE,
    MAX_LENGTH,
    DIALECT_TO_LABEL,
    LABEL_TO_DIALECT,
    NUM_REGIONS,
    ADAPTER_DIM
)

from model import RegionalAdapterWhisper
from data_loader import load_val_split_info, load_test_data
from data_collator import DataCollatorRegionalASR


def load_trained_model(model_dir: str, device: torch.device):
    """
    Load the trained Regional Adapter Whisper model.

    Args:
        model_dir: Path to saved model directory
        device: Device to load model on

    Returns:
        Tuple of (model, processor)
    """
    print(f"Loading model from {model_dir}...")

    # Load processor
    processor = WhisperProcessor.from_pretrained(
        MODEL_NAME, language=LANGUAGE, task=TASK
    )

    # Load adapter config
    adapter_config_path = os.path.join(model_dir, "regional_adapter", "adapter_config.json")
    with open(adapter_config_path, 'r') as f:
        adapter_config = json.load(f)

    num_regions = adapter_config["num_regions"]
    adapter_dim = adapter_config["adapter_dim"]

    # Load base Whisper model
    base_model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME)

    # Create regional model
    regional_model = RegionalAdapterWhisper(
        original_whisper=base_model,
        num_regions=num_regions,
        adapter_dim=adapter_dim
    )

    # Load trained weights
    state_dict = torch.load(
        os.path.join(model_dir, "regional_adapter", "model.pt"),
        map_location=device
    )
    regional_model.load_state_dict(state_dict)

    # Move to device and set to eval mode
    regional_model = regional_model.to(device)
    regional_model.eval()

    print(f"Model loaded successfully")
    return regional_model, processor


def transcribe_and_classify(
    model,
    processor,
    audio_path: str,
    device: torch.device
):
    """
    Transcribe audio and classify dialect.

    Args:
        model: Regional Adapter Whisper model
        processor: Whisper processor
        audio_path: Path to audio file
        device: Device to run inference on

    Returns:
        Tuple of (transcription, predicted_dialect, dialect_confidence, dialect_probs)
    """
    # Load and preprocess audio
    waveform, sr = librosa.load(audio_path, sr=SAMPLING_RATE)

    # Extract features
    input_features = processor.feature_extractor(
        waveform, sampling_rate=SAMPLING_RATE
    ).input_features[0]
    input_features = torch.tensor(input_features).unsqueeze(0).to(device)

    with torch.no_grad():
        # First pass: Get dialect classification
        outputs = model(input_features=input_features)

        if outputs["region_logits"] is not None:
            region_probs = torch.softmax(outputs["region_logits"], dim=-1)
            predicted_dialect_idx = torch.argmax(region_probs, dim=-1).item()
            dialect_confidence = region_probs[0, predicted_dialect_idx].item()
            dialect_probs = region_probs[0].cpu().numpy()
        else:
            predicted_dialect_idx = 0
            dialect_confidence = 0.0
            dialect_probs = np.zeros(model.num_regions)

        # Second pass: Generate transcription with regional adaptation
        region_idx = torch.tensor([predicted_dialect_idx], device=device)

        # Run forward pass with region info to adapt encoder
        _ = model(
            input_features=input_features,
            region_idx=region_idx
        )

        # Generate transcription
        generated_ids = model.whisper.generate(
            input_features,
            max_length=MAX_LENGTH
        )
        transcription = processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0]

    predicted_dialect = LABEL_TO_DIALECT.get(
        predicted_dialect_idx, f"Unknown_{predicted_dialect_idx}"
    )

    return transcription, predicted_dialect, dialect_confidence, dialect_probs


def evaluate_on_validation(
    model,
    processor,
    val_audio: list,
    val_transcriptions: list,
    val_dialects: list,
    device: torch.device,
    output_file: str = None
):
    """
    Evaluate model on validation set.

    Args:
        model: Regional Adapter Whisper model
        processor: Whisper processor
        val_audio: List of validation audio paths
        val_transcriptions: List of ground truth transcriptions
        val_dialects: List of ground truth dialects
        device: Device to run inference on
        output_file: Optional path to save results CSV

    Returns:
        Dictionary of evaluation metrics
    """
    print("\n" + "=" * 80)
    print("Evaluating on Validation Set")
    print("=" * 80)

    wer_metric = evaluate.load("wer")

    results = []
    all_predictions = []
    all_references = []
    dialect_predictions = []
    dialect_references = []

    for i, (audio_path, gt_trans, gt_dialect) in enumerate(
        tqdm(zip(val_audio, val_transcriptions, val_dialects), total=len(val_audio))
    ):
        # Get predictions
        pred_trans, pred_dialect, confidence, probs = transcribe_and_classify(
            model, processor, audio_path, device
        )

        # Store results
        results.append({
            "audio_file": os.path.basename(audio_path),
            "ground_truth_transcription": gt_trans,
            "predicted_transcription": pred_trans,
            "ground_truth_dialect": gt_dialect,
            "predicted_dialect": pred_dialect,
            "dialect_confidence": confidence,
            "dialect_correct": gt_dialect == pred_dialect
        })

        all_predictions.append(pred_trans)
        all_references.append(gt_trans)
        dialect_predictions.append(pred_dialect)
        dialect_references.append(gt_dialect)

    # Calculate metrics
    print("\n" + "=" * 80)
    print("Evaluation Results")
    print("=" * 80)

    # ASR metrics
    wer = wer_metric.compute(predictions=all_predictions, references=all_references)
    print(f"\nASR Word Error Rate (WER): {wer * 100:.2f}%")

    # Dialect classification metrics
    dialect_accuracy = sum(
        1 for p, r in zip(dialect_predictions, dialect_references) if p == r
    ) / len(dialect_predictions)
    print(f"Dialect Classification Accuracy: {dialect_accuracy * 100:.2f}%")

    # Detailed classification report
    print("\nDialect Classification Report:")
    print(classification_report(
        dialect_references, dialect_predictions,
        target_names=list(DIALECT_TO_LABEL.keys())
    ))

    # Confusion matrix
    print("\nConfusion Matrix:")
    cm = confusion_matrix(
        dialect_references, dialect_predictions,
        labels=list(DIALECT_TO_LABEL.keys())
    )
    print(pd.DataFrame(
        cm,
        index=list(DIALECT_TO_LABEL.keys()),
        columns=list(DIALECT_TO_LABEL.keys())
    ))

    # Save results to CSV
    if output_file:
        df = pd.DataFrame(results)
        df.to_csv(output_file, index=False, encoding='utf-8')
        print(f"\nResults saved to {output_file}")

    # Per-dialect metrics
    print("\n" + "-" * 40)
    print("Per-Dialect Metrics:")
    dialect_wers = defaultdict(list)
    for result in results:
        dialect = result["ground_truth_dialect"]
        sample_wer = wer_metric.compute(
            predictions=[result["predicted_transcription"]],
            references=[result["ground_truth_transcription"]]
        )
        dialect_wers[dialect].append(sample_wer)

    for dialect in sorted(dialect_wers.keys()):
        avg_wer = np.mean(dialect_wers[dialect])
        count = len(dialect_wers[dialect])
        print(f"  {dialect}: WER = {avg_wer * 100:.2f}% (n={count})")

    return {
        "wer": wer,
        "dialect_accuracy": dialect_accuracy,
        "results": results
    }


def inference_on_test(
    model,
    processor,
    test_audio: list,
    device: torch.device,
    output_file: str
):
    """
    Run inference on test set (no ground truth).

    Args:
        model: Regional Adapter Whisper model
        processor: Whisper processor
        test_audio: List of test audio paths
        device: Device to run inference on
        output_file: Path to save results CSV
    """
    print("\n" + "=" * 80)
    print("Running Inference on Test Set")
    print("=" * 80)

    results = []

    for audio_path in tqdm(test_audio, desc="Processing test files"):
        # Get predictions
        transcription, dialect, confidence, probs = transcribe_and_classify(
            model, processor, audio_path, device
        )

        results.append({
            "audio_file": os.path.basename(audio_path),
            "predicted_transcription": transcription,
            "predicted_dialect": dialect,
            "dialect_confidence": confidence
        })

    # Save results
    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False, encoding='utf-8')
    print(f"\nTest results saved to {output_file}")

    # Summary
    print("\nDialect Distribution in Predictions:")
    dialect_counts = df["predicted_dialect"].value_counts()
    for dialect, count in dialect_counts.items():
        print(f"  {dialect}: {count} ({count/len(df)*100:.1f}%)")


def main():
    """Main inference function."""
    import argparse

    parser = argparse.ArgumentParser(description="Tamil Dialect Classification Inference")
    parser.add_argument(
        "--mode", type=str, choices=["val", "test"], default="val",
        help="Inference mode: 'val' for validation, 'test' for test set"
    )
    parser.add_argument(
        "--model_dir", type=str, default=OUTPUT_DIR,
        help="Path to trained model directory"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output CSV file path"
    )
    args = parser.parse_args()

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    model, processor = load_trained_model(args.model_dir, device)

    if args.mode == "val":
        # Load validation data
        val_split_path = os.path.join(args.model_dir, "val_split.json")
        val_audio, val_trans, val_dialects = load_val_split_info(val_split_path)

        # Evaluate
        output_file = args.output or os.path.join(args.model_dir, "val_results.csv")
        evaluate_on_validation(
            model, processor,
            val_audio, val_trans, val_dialects,
            device,
            output_file
        )

    elif args.mode == "test":
        # Load test data
        test_audio = load_test_data(TEST_DIR)

        # Run inference
        output_file = args.output or os.path.join(args.model_dir, "test_predictions.csv")
        inference_on_test(model, processor, test_audio, device, output_file)


if __name__ == "__main__":
    main()
