"""
t-SNE Visualization of Whisper Encoder Regional Embeddings
Generates a t-SNE plot showing how the trained encoder clusters dialect embeddings.

Uses the attention-pooled encoder output (the same representation fed to the dialect classifier)
to visualize regional clustering in the learned embedding space.
"""
import os
import time
import json
import argparse
import torch
import numpy as np
import librosa
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from config import (
    MODEL_NAME, TRAIN_DIR, DIALECT_DIRS, DIALECT_TO_LABEL,
    LABEL_TO_DIALECT, SAMPLING_RATE, OUTPUT_DIR, SEED, NUM_REGIONS, ADAPTER_DIM
)
from model import RegionalAdapterWhisper
from data_loader import load_dialect_data
from tamil_text_normalizer import create_normalizer


def parse_args():
    parser = argparse.ArgumentParser(description="t-SNE visualization of dialect embeddings")
    parser.add_argument("--train_dir", type=str, default=TRAIN_DIR,
                        help="Path to training data directory")
    parser.add_argument("--model_dir", type=str, default=OUTPUT_DIR,
                        help="Path to saved model directory (contains regional_adapter/)")
    parser.add_argument("--output_dir", type=str, default="/kaggle/working",
                        help="Directory to save plot and embeddings")
    parser.add_argument("--max_per_dialect", type=int, default=None,
                        help="Max samples per dialect (None = use all)")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for embedding extraction")
    parser.add_argument("--perplexity", type=int, default=30,
                        help="t-SNE perplexity")
    return parser.parse_args()


# t-SNE hyperparameters (defaults, overridden by args)
TSNE_N_ITER = 1000
TSNE_LEARNING_RATE = "auto"

# Dialect colors and markers
DIALECT_COLORS = {
    "Northern_Dialect": "#e74c3c",
    "Southern_Dialect": "#2ecc71",
    "Western_Dialect": "#3498db",
    "Central_Dialect": "#f39c12",
}
DIALECT_MARKERS = {
    "Northern_Dialect": "o",
    "Southern_Dialect": "s",
    "Western_Dialect": "^",
    "Central_Dialect": "D",
}


def log_time(message, start_time):
    """Log elapsed time since start."""
    elapsed = time.time() - start_time
    hours, rem = divmod(elapsed, 3600)
    minutes, seconds = divmod(rem, 60)
    if hours > 0:
        print(f"[TIME] {message}: {int(hours)}h {int(minutes)}m {seconds:.1f}s")
    elif minutes > 0:
        print(f"[TIME] {message}: {int(minutes)}m {seconds:.1f}s")
    else:
        print(f"[TIME] {message}: {seconds:.1f}s")
    return time.time()


def load_model(adapter_path, device):
    """Load the trained RegionalAdapterWhisper model."""
    print(f"Loading base Whisper model: {MODEL_NAME}")
    base_whisper = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME)
    processor = WhisperProcessor.from_pretrained(MODEL_NAME)

    print(f"Loading trained adapter from: {adapter_path}")
    model = RegionalAdapterWhisper.from_pretrained(adapter_path, base_whisper)
    model = model.to(device)
    model.eval()

    return model, processor


def extract_embeddings(model, processor, audio_paths, dialects, device, batch_size=32):
    """
    Extract attention-pooled encoder embeddings for all audio samples.

    This extracts the same pooled representation that feeds into the dialect
    classifier — the learned embedding space where dialects should cluster.
    """
    all_embeddings = []
    all_labels = []
    all_paths = []

    num_samples = len(audio_paths)
    print(f"\nExtracting embeddings for {num_samples} samples (batch_size={batch_size})...")

    for batch_start in range(0, num_samples, batch_size):
        batch_end = min(batch_start + batch_size, num_samples)
        batch_audio_paths = audio_paths[batch_start:batch_end]
        batch_dialects = dialects[batch_start:batch_end]

        # Load and process audio
        batch_features = []
        valid_indices = []

        for i, audio_path in enumerate(batch_audio_paths):
            try:
                audio, sr = librosa.load(audio_path, sr=SAMPLING_RATE)
                features = processor.feature_extractor(
                    audio, sampling_rate=SAMPLING_RATE, return_tensors="pt"
                )
                batch_features.append(features.input_features.squeeze(0))
                valid_indices.append(i)
            except Exception as e:
                print(f"  Warning: Failed to load {os.path.basename(audio_path)}: {e}")

        if not batch_features:
            continue

        # Pad to same length and stack
        max_len = max(f.shape[-1] for f in batch_features)
        padded = []
        for f in batch_features:
            if f.shape[-1] < max_len:
                pad_size = max_len - f.shape[-1]
                f = torch.nn.functional.pad(f, (0, pad_size))
            padded.append(f)

        input_features = torch.stack(padded).to(device)

        # Forward through encoder + attention pooling (no decoder needed)
        with torch.no_grad():
            encoder_outputs = model.whisper.model.encoder(input_features=input_features)
            hidden_states = encoder_outputs.last_hidden_state

            # Attention pooling — same as in model.forward()
            attn_scores = model.attention_pool(hidden_states)
            attn_weights = torch.nn.functional.softmax(attn_scores, dim=1)
            pooled_output = (attn_weights * hidden_states).sum(dim=1)  # [batch, d_model]

        embeddings = pooled_output.cpu().numpy()
        all_embeddings.append(embeddings)

        for idx in valid_indices:
            all_labels.append(batch_dialects[idx])
            all_paths.append(batch_audio_paths[idx])

        # Progress
        processed = min(batch_end, num_samples)
        if processed % (batch_size * 5) == 0 or processed == num_samples:
            print(f"  Processed {processed}/{num_samples} samples")

    all_embeddings = np.concatenate(all_embeddings, axis=0)
    print(f"  Extracted embeddings shape: {all_embeddings.shape}")
    return all_embeddings, all_labels, all_paths


def run_tsne(embeddings, perplexity=30, n_iter=1000, learning_rate="auto", seed=42):
    """Run t-SNE dimensionality reduction."""
    print(f"\nRunning t-SNE (perplexity={perplexity}, n_iter={n_iter})...")
    n_samples = embeddings.shape[0]

    # Adjust perplexity if too large for the number of samples
    effective_perplexity = min(perplexity, max(5, n_samples // 4))
    if effective_perplexity != perplexity:
        print(f"  Adjusted perplexity to {effective_perplexity} (n_samples={n_samples})")

    tsne = TSNE(
        n_components=2,
        perplexity=effective_perplexity,
        n_iter=n_iter,
        learning_rate=learning_rate,
        random_state=seed,
        init="pca",
        metric="cosine",
    )
    tsne_results = tsne.fit_transform(embeddings)
    print(f"  t-SNE complete. KL divergence: {tsne.kl_divergence_:.4f}")
    return tsne_results


def create_tsne_plot(tsne_results, labels, save_path):
    """Create and save the t-SNE visualization."""
    print(f"\nGenerating t-SNE plot...")

    fig, ax = plt.subplots(1, 1, figsize=(12, 10))

    # Plot each dialect
    unique_dialects = sorted(set(labels))
    for dialect in unique_dialects:
        mask = np.array([l == dialect for l in labels])
        display_name = dialect.replace("_", " ")
        count = mask.sum()

        ax.scatter(
            tsne_results[mask, 0],
            tsne_results[mask, 1],
            c=DIALECT_COLORS.get(dialect, "#999999"),
            marker=DIALECT_MARKERS.get(dialect, "o"),
            label=f"{display_name} (n={count})",
            alpha=0.65,
            s=50,
            edgecolors="white",
            linewidths=0.5,
        )

    ax.set_title(
        "t-SNE of Whisper Encoder Dialect Embeddings\n"
        "(Attention-Pooled Encoder Representations)",
        fontsize=14,
        fontweight="bold",
        pad=15,
    )
    ax.set_xlabel("t-SNE Dimension 1", fontsize=12)
    ax.set_ylabel("t-SNE Dimension 2", fontsize=12)
    ax.legend(
        fontsize=11,
        loc="best",
        framealpha=0.9,
        edgecolor="gray",
        markerscale=1.3,
    )
    ax.grid(True, alpha=0.2, linestyle="--")
    ax.set_axisbelow(True)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Plot saved to: {save_path}")


def main():
    args = parse_args()

    adapter_path = os.path.join(args.model_dir, "regional_adapter")
    plot_save_path = os.path.join(args.output_dir, "tsne_dialect_embeddings.png")
    embeddings_save_path = os.path.join(args.output_dir, "dialect_embeddings.npz")

    total_start = time.time()
    print("=" * 70)
    print("  t-SNE Visualization of Regional Dialect Embeddings")
    print("=" * 70)
    print(f"\n  Train dir:  {args.train_dir}")
    print(f"  Model dir:  {args.model_dir}")
    print(f"  Output dir: {args.output_dir}")

    # ---- Device setup ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")

    # ---- Load model ----
    step_start = time.time()
    model, processor = load_model(adapter_path, device)
    step_start = log_time("Model loading", step_start)

    # ---- Load dataset ----
    print("\n" + "-" * 50)
    normalizer = create_normalizer("default")
    audio_paths, transcriptions, dialects = load_dialect_data(
        args.train_dir, DIALECT_DIRS, normalizer
    )
    step_start = log_time("Data loading", step_start)

    # ---- Optional subsetting ----
    if args.max_per_dialect is not None:
        print(f"\nSubsetting to max {args.max_per_dialect} samples per dialect...")
        from collections import defaultdict
        dialect_indices = defaultdict(list)
        for i, d in enumerate(dialects):
            dialect_indices[d].append(i)

        selected_indices = []
        rng = np.random.RandomState(SEED)
        for dialect, indices in dialect_indices.items():
            if len(indices) > args.max_per_dialect:
                indices = rng.choice(indices, args.max_per_dialect, replace=False).tolist()
            selected_indices.extend(indices)
        selected_indices.sort()

        audio_paths = [audio_paths[i] for i in selected_indices]
        transcriptions = [transcriptions[i] for i in selected_indices]
        dialects = [dialects[i] for i in selected_indices]
        print(f"  Using {len(audio_paths)} samples total")

    # Print dialect distribution
    from collections import Counter
    dist = Counter(dialects)
    print("\nDialect distribution:")
    for dialect, count in sorted(dist.items()):
        print(f"  {dialect}: {count}")

    # ---- Extract embeddings ----
    print("\n" + "-" * 50)
    embeddings, labels, paths = extract_embeddings(
        model, processor, audio_paths, dialects, device,
        batch_size=args.batch_size
    )
    step_start = log_time("Embedding extraction", step_start)

    # ---- Save raw embeddings ----
    os.makedirs(args.output_dir, exist_ok=True)
    np.savez_compressed(
        embeddings_save_path,
        embeddings=embeddings,
        labels=np.array(labels),
        paths=np.array(paths),
    )
    print(f"  Embeddings saved to: {embeddings_save_path}")

    # ---- Run t-SNE ----
    print("\n" + "-" * 50)
    tsne_results = run_tsne(
        embeddings,
        perplexity=args.perplexity,
        n_iter=TSNE_N_ITER,
        learning_rate=TSNE_LEARNING_RATE,
        seed=SEED,
    )
    step_start = log_time("t-SNE computation", step_start)

    # ---- Generate plot ----
    print("\n" + "-" * 50)
    create_tsne_plot(tsne_results, labels, plot_save_path)
    step_start = log_time("Plot generation", step_start)

    # ---- Summary ----
    print("\n" + "=" * 70)
    log_time("TOTAL t-SNE pipeline", total_start)
    print(f"  Samples visualized: {len(labels)}")
    print(f"  Embedding dim: {embeddings.shape[1]}")
    print(f"  Plot: {plot_save_path}")
    print(f"  Embeddings: {embeddings_save_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
