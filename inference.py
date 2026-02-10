"""
Inference Script for Tamil Dialect Classification + ASR - Competition Submission
Generates submission files for both subtasks in the required format
Subtask 1: Speech-Based Dialect Classification
Subtask 2: Automatic Speech Recognition (ASR) for Dialectal Tamil
"""
import os
import json
import torch
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score
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
    Transcribe audio and classify dialect (both tasks).

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
        # First: Get dialect classification
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

        # Second: Generate transcription with regional adaptation
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


def convert_dialect_to_submission_format(dialect_name: str) -> str:
    """
    Convert internal dialect name to submission format.
    
    Args:
        dialect_name: Internal dialect name (e.g., "Northern_Dialect")
    
    Returns:
        Submission format dialect code (e.g., "Northern")
    """
    # Map from internal names to submission format
    dialect_mapping = {
        "Northern_Dialect": "Northern",
        "Southern_Dialect": "Southern",
        "Western_Dialect": "Western",
        "Central_Dialect": "Central"
    }
    return dialect_mapping.get(dialect_name, dialect_name)


def generate_classification_submission(
    model,
    processor,
    test_audio: list,
    device: torch.device,
    output_file: str,
    team_name: str = "TEAM",
    run_number: int = 1
):
    """
    Generate classification submission file in required format.
    
    Format: <test_file_id> <dialect_code>
    Example:
        test_0001 Central
        test_0002 Southern
        test_0003 Northern
        test_0004 Western

    Args:
        model: Regional Adapter Whisper model
        processor: Whisper processor
        test_audio: List of test audio paths
        device: Device to run inference on
        output_file: Path to save submission file
        team_name: Team name for submission
        run_number: Run number (1, 2, or 3)
    """
    print("\n" + "=" * 80)
    print(f"Generating Classification Submission - Run {run_number}")
    print("=" * 80)

    results = []
    
    for audio_path in tqdm(test_audio, desc="Processing test files"):
        # Get prediction (both transcription and classification)
        transcription, dialect, confidence, probs = transcribe_and_classify(
            model, processor, audio_path, device
        )
        
        # Extract test file ID from filename (e.g., test_0001.wav -> test_0001)
        file_name = os.path.basename(audio_path)
        test_file_id = os.path.splitext(file_name)[0]
        
        # Convert to submission format
        dialect_code = convert_dialect_to_submission_format(dialect)
        
        results.append({
            "test_file_id": test_file_id,
            "dialect_code": dialect_code,
            "transcription": transcription,
            "confidence": confidence
        })
    
    # Sort by test file ID to ensure consistent ordering
    results.sort(key=lambda x: x["test_file_id"])
    
    # Write to submission file
    with open(output_file, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(f"{result['test_file_id']} {result['dialect_code']}\n")
    
    print(f"\nClassification submission file saved to: {output_file}")
    print(f"Format: {team_name}_Classification_Run{run_number}.txt")
    
    # Summary statistics
    print("\nDialect Distribution in Predictions:")
    dialect_counts = defaultdict(int)
    for result in results:
        dialect_counts[result["dialect_code"]] += 1
    
    total = len(results)
    for dialect in sorted(dialect_counts.keys()):
        count = dialect_counts[dialect]
        print(f"  {dialect}: {count} ({count/total*100:.1f}%)")
    
    print(f"\nTotal predictions: {total}")
    
    return results


def generate_recognition_submission(
    model,
    processor,
    test_audio: list,
    device: torch.device,
    output_file: str,
    team_name: str = "TEAM",
    run_number: int = 1
):
    """
    Generate ASR (recognition) submission file in required format.
    
    Format: <test_file_id> <recognized_text>
    Example:
        test_0001 சொல்லுங்கோ
        test_0002 எனக்கு உதவி வேண்டும்
        test_0003 வங்கிக்கு போகணும்

    Args:
        model: Regional Adapter Whisper model
        processor: Whisper processor
        test_audio: List of test audio paths
        device: Device to run inference on
        output_file: Path to save submission file
        team_name: Team name for submission
        run_number: Run number (1, 2, or 3)
    """
    print("\n" + "=" * 80)
    print(f"Generating Recognition (ASR) Submission - Run {run_number}")
    print("=" * 80)

    results = []
    
    for audio_path in tqdm(test_audio, desc="Transcribing test files"):
        # Get prediction (both transcription and classification)
        transcription, dialect, confidence, probs = transcribe_and_classify(
            model, processor, audio_path, device
        )
        
        # Extract test file ID from filename (e.g., test_0001.wav -> test_0001)
        file_name = os.path.basename(audio_path)
        test_file_id = os.path.splitext(file_name)[0]
        
        results.append({
            "test_file_id": test_file_id,
            "transcription": transcription,
            "dialect": dialect,
            "confidence": confidence
        })
    
    # Sort by test file ID to ensure consistent ordering
    results.sort(key=lambda x: x["test_file_id"])
    
    # Write to submission file
    with open(output_file, 'w', encoding='utf-8') as f:
        for result in results:
            # Clean transcription (remove extra spaces)
            clean_transcription = ' '.join(result['transcription'].split())
            f.write(f"{result['test_file_id']} {clean_transcription}\n")
    
    print(f"\nRecognition submission file saved to: {output_file}")
    print(f"Format: {team_name}_Recognition_Run{run_number}.txt")
    print(f"\nTotal transcriptions: {len(results)}")
    
    # Show sample transcriptions
    print("\nSample transcriptions:")
    for i in range(min(5, len(results))):
        print(f"  {results[i]['test_file_id']}: {results[i]['transcription'][:50]}...")
    
    return results


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
    Evaluate both classification and ASR performance on validation set.
    Computes:
    - Classification: macro-averaged precision, recall, F1 score
    - ASR: Word Error Rate (WER)

    Args:
        model: Regional Adapter Whisper model
        processor: Whisper processor
        val_audio: List of validation audio paths
        val_transcriptions: List of ground truth transcriptions
        val_dialects: List of ground truth dialects
        device: Device to run inference on
        output_file: Optional path to save detailed results CSV
    """
    print("\n" + "=" * 80)
    print("Evaluating on Validation Set (Both Tasks)")
    print("=" * 80)

    wer_metric = evaluate.load("wer")

    results = []
    all_predictions = []
    all_references = []
    dialect_predictions = []
    dialect_references = []

    for audio_path, gt_trans, gt_dialect in tqdm(
        zip(val_audio, val_transcriptions, val_dialects), 
        total=len(val_audio),
        desc="Processing validation files"
    ):
        # Get predictions (both transcription and classification)
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

    # =========================================================================
    # SUBTASK 1: CLASSIFICATION METRICS
    # =========================================================================
    print("\n" + "=" * 80)
    print("SUBTASK 1: Dialect Classification Results")
    print("=" * 80)

    # Overall accuracy
    dialect_accuracy = sum(
        1 for p, r in zip(dialect_predictions, dialect_references) if p == r
    ) / len(dialect_predictions)
    print(f"\nOverall Accuracy: {dialect_accuracy * 100:.2f}%")

    # Macro-averaged metrics (as required by competition)
    macro_precision = precision_score(
        dialect_references, dialect_predictions,
        labels=list(DIALECT_TO_LABEL.keys()),
        average='macro',
        zero_division=0
    )
    macro_recall = recall_score(
        dialect_references, dialect_predictions,
        labels=list(DIALECT_TO_LABEL.keys()),
        average='macro',
        zero_division=0
    )
    macro_f1 = f1_score(
        dialect_references, dialect_predictions,
        labels=list(DIALECT_TO_LABEL.keys()),
        average='macro',
        zero_division=0
    )

    print(f"\n*** Competition Metrics (Classification) ***")
    print(f"Macro-averaged Precision: {macro_precision * 100:.2f}%")
    print(f"Macro-averaged Recall:    {macro_recall * 100:.2f}%")
    print(f"Macro-averaged F1 Score:  {macro_f1 * 100:.2f}%")

    # Detailed classification report
    print("\n" + "-" * 80)
    print("Detailed Classification Report:")
    print("-" * 80)
    print(classification_report(
        dialect_references, dialect_predictions,
        target_names=list(DIALECT_TO_LABEL.keys()),
        digits=4,
        zero_division=0
    ))

    # Confusion matrix
    print("\n" + "-" * 80)
    print("Confusion Matrix:")
    print("-" * 80)
    cm = confusion_matrix(
        dialect_references, dialect_predictions,
        labels=list(DIALECT_TO_LABEL.keys())
    )
    cm_df = pd.DataFrame(
        cm,
        index=list(DIALECT_TO_LABEL.keys()),
        columns=list(DIALECT_TO_LABEL.keys())
    )
    print(cm_df)

    # =========================================================================
    # SUBTASK 2: ASR METRICS
    # =========================================================================
    print("\n" + "=" * 80)
    print("SUBTASK 2: ASR (Recognition) Results")
    print("=" * 80)

    # Overall WER
    wer = wer_metric.compute(predictions=all_predictions, references=all_references)
    print(f"\n*** Competition Metric (ASR) ***")
    print(f"Word Error Rate (WER): {wer * 100:.2f}%")

    # Per-dialect WER
    print("\n" + "-" * 80)
    print("Per-Dialect WER:")
    print("-" * 80)
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

    # =========================================================================
    # COMBINED SUMMARY
    # =========================================================================
    print("\n" + "=" * 80)
    print("COMBINED SUMMARY")
    print("=" * 80)
    print(f"Classification F1:  {macro_f1 * 100:.2f}%")
    print(f"ASR WER:            {wer * 100:.2f}%")
    print(f"Combined Score:     {((macro_f1 * 100) + (100 - wer * 100)) / 2:.2f}%")

    # Save results to CSV if requested
    if output_file:
        df = pd.DataFrame(results)
        df.to_csv(output_file, index=False, encoding='utf-8')
        print(f"\nDetailed results saved to {output_file}")

    return {
        "accuracy": dialect_accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "wer": wer,
        "results": results
    }


def main():
    """Main inference function for generating submission files for both subtasks."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Tamil Dialect Classification + ASR - Generate Submission Files"
    )
    parser.add_argument(
        "--mode", type=str, 
        choices=["val", "test"], 
        required=True,
        help="Inference mode: 'val' for validation evaluation, 'test' for test submission"
    )
    parser.add_argument(
        "--subtask", type=str,
        choices=["classification", "recognition", "both"],
        default="both",
        help="Which subtask to run: 'classification' (Subtask 1), 'recognition' (Subtask 2), or 'both'"
    )
    parser.add_argument(
        "--model_dir", type=str, 
        default=OUTPUT_DIR,
        help="Path to trained model directory"
    )
    parser.add_argument(
        "--test_dir", type=str,
        default=None,
        help="Path to test audio directory (overrides TEST_DIR from config)"
    )
    parser.add_argument(
        "--output_dir", type=str, 
        default=None,
        help="Output directory for submission files (default: model_dir)"
    )
    parser.add_argument(
        "--team_name", type=str, 
        default="TEAM",
        help="Team name for submission files"
    )
    parser.add_argument(
        "--run_number", type=int, 
        default=1, 
        choices=[1, 2, 3],
        help="Run number (1, 2, or 3)"
    )
    args = parser.parse_args()

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Set output directory
    output_dir = args.output_dir or args.model_dir

    # Load model
    model, processor = load_trained_model(args.model_dir, device)

    if args.mode == "val":
        # =====================================================================
        # VALIDATION MODE: Evaluate on validation set
        # =====================================================================
        print("\n" + "=" * 80)
        print("MODE: Validation Evaluation")
        print("=" * 80)
        
        val_split_path = os.path.join(args.model_dir, "val_split.json")
        val_audio, val_trans, val_dialects = load_val_split_info(val_split_path)

        # Evaluate both tasks
        output_file = os.path.join(
            output_dir, 
            f"val_results_run{args.run_number}.csv"
        )
        
        metrics = evaluate_on_validation(
            model, processor,
            val_audio, val_trans, val_dialects,
            device,
            output_file
        )
        
        print("\n" + "=" * 80)
        print("FINAL SUMMARY - Validation Results")
        print("=" * 80)
        print(f"\nSubtask 1: Classification")
        print(f"  Accuracy:          {metrics['accuracy'] * 100:.2f}%")
        print(f"  Macro Precision:   {metrics['macro_precision'] * 100:.2f}%")
        print(f"  Macro Recall:      {metrics['macro_recall'] * 100:.2f}%")
        print(f"  Macro F1 Score:    {metrics['macro_f1'] * 100:.2f}%")
        print(f"\nSubtask 2: ASR")
        print(f"  Word Error Rate:   {metrics['wer'] * 100:.2f}%")
        print(f"\nCombined Performance: {((metrics['macro_f1'] * 100) + (100 - metrics['wer'] * 100)) / 2:.2f}%")

    elif args.mode == "test":
        # =====================================================================
        # TEST MODE: Generate submission files
        # =====================================================================
        print("\n" + "=" * 80)
        print("MODE: Test Submission Generation")
        print("=" * 80)
        
        # Determine test directory
        test_dir = args.test_dir or TEST_DIR
        print(f"Test directory: {test_dir}")
        
        # Load test data
        test_audio = load_test_data(test_dir)
        print(f"Found {len(test_audio)} test files")
        
        # Generate submission files based on subtask
        if args.subtask in ["classification", "both"]:
            print("\n" + "=" * 80)
            print("Generating SUBTASK 1: Classification Submission")
            print("=" * 80)
            
            classification_output = os.path.join(
                output_dir,
                f"{args.team_name}_Classification_Run{args.run_number}.txt"
            )
            
            classification_results = generate_classification_submission(
                model, processor, 
                test_audio, device, 
                classification_output,
                args.team_name, 
                args.run_number
            )
        
        if args.subtask in ["recognition", "both"]:
            print("\n" + "=" * 80)
            print("Generating SUBTASK 2: Recognition (ASR) Submission")
            print("=" * 80)
            
            recognition_output = os.path.join(
                output_dir,
                f"{args.team_name}_Recognition_Run{args.run_number}.txt"
            )
            
            recognition_results = generate_recognition_submission(
                model, processor, 
                test_audio, device, 
                recognition_output,
                args.team_name, 
                args.run_number
            )
        
        # Final summary
        print("\n" + "=" * 80)
        print("Submission files generated successfully!")
        print("=" * 80)
        if args.subtask in ["classification", "both"]:
            print(f"Classification: {classification_output}")
        if args.subtask in ["recognition", "both"]:
            print(f"Recognition:    {recognition_output}")
        print(f"\nTotal test files processed: {len(test_audio)}")
        print("\nNext steps:")
        print("1. Verify the format of both submission files")
        print("2. Package both files into a ZIP file")
        print("3. Submit to the competition")


if __name__ == "__main__":
    main()