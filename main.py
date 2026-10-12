"""
main.py -- SVD-Based Image Compression & Classification Experiment
==================================================================
3rd-Semester Mathematical Foundations of AI & Data Science Mini-Project

This script runs two classification experiments on the same dataset:

  Experiment A: Original grayscale images → SVM classifier
  Experiment B: SVD-compressed images     → SVM classifier

Both experiments use the exact same:
  - Train/test split  (stratified, fixed random seed)
  - Preprocessing     (grayscale, resize to 128×128)
  - Classifier        (linear SVM with StandardScaler)
  - Evaluation metric (accuracy)

The ONLY difference is whether the images are compressed via truncated SVD.

Usage:
    python main.py

All results are saved to the results/ directory.
"""

import os
import sys

# Configure UTF-8 output on Windows to prevent UnicodeEncodeError
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import numpy as np
from sklearn.model_selection import train_test_split

# ===========================================================================
# CONFIGURABLE PARAMETERS -- Change these to experiment!
# ===========================================================================
SVD_RANK_K = 50                        # Number of singular values to retain
ANALYSIS_RANKS = [5, 10, 20, 30, 50]   # Ranks to test in the rank-analysis plots
MAX_IMAGES_PER_CLASS = 350             # 350 cats + 350 dogs = 700 images total (None for all)
CLASSIFIER_KERNEL = "rbf"              # SVM kernel: 'rbf' or 'linear'
TEST_SIZE = 0.25                       # Fraction of data used for testing
RANDOM_SEED = 42                       # Fixed seed for reproducibility
DATASET_DIR = "dataset"                # Path to the image dataset
RESULTS_DIR = "results"                # Path to save outputs
# ===========================================================================

# Local imports
from src.preprocessing import load_dataset, flatten_images, IMG_HEIGHT, IMG_WIDTH
from src.svd_compression import (
    compress_image_batch,
    compute_compression_ratio,
)
from src.classifier import train_and_evaluate
from src.experiment import (
    run_rank_analysis,
    plot_accuracy_comparison,
    save_visual_comparison,
)


def main():
    # Ensure reproducibility
    np.random.seed(RANDOM_SEED)

    # Create results directory
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ===================================================================
    # STEP 1: Load and preprocess the dataset
    # ===================================================================
    print("=" * 60)
    print("STEP 1: Loading and preprocessing dataset")
    print("=" * 60)

    images, labels, class_names = load_dataset(
        DATASET_DIR, max_images_per_class=MAX_IMAGES_PER_CLASS
    )
    labels = np.array(labels)

    num_images = len(images)
    m, n = images[0].shape  # Should be (128, 128)

    print(f"\nTotal images loaded: {num_images}")
    print(f"Image dimensions:   {m} x {n}  (each image is a matrix A in R^({m}x{n}))")
    print(f"Classes:            {class_names}")

    # ===================================================================
    # STEP 2: Train/test split (SAME split for both experiments)
    # ===================================================================
    print("\n" + "=" * 60)
    print("STEP 2: Splitting data into train and test sets")
    print("=" * 60)

    # Use list indices to split, so we can apply the same split to
    # both original and compressed images
    indices = np.arange(num_images)
    idx_train, idx_test = train_test_split(
        indices, test_size=TEST_SIZE, random_state=RANDOM_SEED,
        stratify=labels  # Ensure both classes are represented equally
    )

    train_images = [images[i] for i in idx_train]
    test_images = [images[i] for i in idx_test]
    y_train = labels[idx_train]
    y_test = labels[idx_test]

    print(f"Training set: {len(train_images)} images")
    print(f"Test set:     {len(test_images)} images")
    print(f"Test size:    {TEST_SIZE * 100:.0f}%")

    # ===================================================================
    # STEP 3: Applying SVD compression of images
    # ===================================================================
    print("\n" + "=" * 60)
    print(f"STEP 3: Applying SVD compression (rank k = {SVD_RANK_K})")
    print("=" * 60)
    print(f"  Each image A in R^({m}x{n}) is decomposed as A = U Sigma V^T")
    print(f"  Keeping only the first k={SVD_RANK_K} singular values/vectors")
    print(f"  Reconstructing A_k = U_k Sigma_k V_k^T")

    # Compress training images
    print("\n  Compressing training images...")
    train_compressed, train_errors, train_energies = compress_image_batch(
        train_images, SVD_RANK_K
    )

    # Compress test images
    print("  Compressing test images...")
    test_compressed, test_errors, test_energies = compress_image_batch(
        test_images, SVD_RANK_K
    )

    avg_train_error = np.mean(train_errors)
    avg_test_error = np.mean(test_errors)
    avg_energy = np.mean(train_energies + test_energies)
    compression_ratio = compute_compression_ratio(m, n, SVD_RANK_K)

    print(f"\n  Average reconstruction error (train): {avg_train_error:.2f}")
    print(f"  Average reconstruction error (test):  {avg_test_error:.2f}")
    print(f"  Average energy retained:              {avg_energy:.2f}%")
    print(f"  Theoretical compression ratio:        {compression_ratio:.2f}x")
    print(f"    Original storage:  {m} × {n} = {m*n} values")
    print(f"    Truncated storage: {SVD_RANK_K} × ({m} + {n} + 1) "
          f"= {SVD_RANK_K * (m + n + 1)} values")

    # ===================================================================
    # STEP 4: EXPERIMENT A — Classification on ORIGINAL images
    # ===================================================================
    print("\n" + "=" * 60)
    print("STEP 4: EXPERIMENT A — Original Images → SVM Classifier")
    print("=" * 60)
    print("  Pipeline: image → grayscale → resize → flatten → SVM")

    X_train_orig = flatten_images(train_images)
    X_test_orig = flatten_images(test_images)

    print(f"  Feature vector length: {X_train_orig.shape[1]} "
          f"({m}×{n} flattened)")

    acc_original, _ = train_and_evaluate(
        X_train_orig, y_train, X_test_orig, y_test, kernel=CLASSIFIER_KERNEL
    )
    print(f"\n  [+] Original accuracy: {acc_original * 100:.2f}%")

    # ===================================================================
    # STEP 5: EXPERIMENT B — Classification on SVD-COMPRESSED images
    # ===================================================================
    print("\n" + "=" * 60)
    print(f"STEP 5: EXPERIMENT B — SVD Compressed (k={SVD_RANK_K}) → SVM Classifier")
    print("=" * 60)
    print(f"  Pipeline: image → grayscale → resize → SVD (k={SVD_RANK_K}) "
          f"→ reconstruct → flatten → SVM")

    X_train_comp = flatten_images(train_compressed)
    X_test_comp = flatten_images(test_compressed)

    acc_compressed, _ = train_and_evaluate(
        X_train_comp, y_train, X_test_comp, y_test, kernel=CLASSIFIER_KERNEL
    )
    print(f"\n  [+] Compressed accuracy: {acc_compressed * 100:.2f}%")

    # ===================================================================
    # STEP 6: Results comparison
    # ===================================================================
    acc_diff = (acc_original - acc_compressed) * 100

    print("\n" + "=" * 60)
    print("STEP 6: Results Comparison")
    print("=" * 60)
    print(f"  Original accuracy:   {acc_original * 100:.2f}%")
    print(f"  Compressed accuracy: {acc_compressed * 100:.2f}%")
    print(f"  Accuracy difference: {acc_diff:+.2f} percentage points")

    if abs(acc_diff) < 2.0:
        interpretation = (
            "The SVD compression caused NEGLIGIBLE change in accuracy. "
            "The classifier performs essentially the same on compressed images."
        )
    elif acc_diff > 0:
        interpretation = (
            f"The SVD compression DECREASED accuracy by {abs(acc_diff):.1f} "
            f"percentage points. Some discriminative information was lost "
            f"during compression."
        )
    else:
        interpretation = (
            f"The SVD compression actually INCREASED accuracy by "
            f"{abs(acc_diff):.1f} percentage points. This may indicate "
            f"that SVD acted as a denoising step, removing noise that "
            f"confused the classifier."
        )
    print(f"\n  Interpretation: {interpretation}")

    # ===================================================================
    # STEP 7: Generate plots and visual outputs
    # ===================================================================
    print("\n" + "=" * 60)
    print("STEP 7: Generating plots and visual outputs")
    print("=" * 60)

    # Accuracy comparison bar chart
    plot_accuracy_comparison(acc_original, acc_compressed, SVD_RANK_K,
                            RESULTS_DIR)

    # Rank analysis plots (error, energy, compression ratio vs rank)
    all_images = train_images + test_images
    rank_errors, rank_energies, rank_ratios = run_rank_analysis(
        all_images, ANALYSIS_RANKS, RESULTS_DIR
    )

    # Visual comparison of SVD reconstructions
    # Use the first test image for demonstration
    sample_image = test_images[0]
    save_visual_comparison(sample_image, ANALYSIS_RANKS, RESULTS_DIR)

    # ===================================================================
    # FINAL SUMMARY
    # ===================================================================
    avg_all_error = np.mean(train_errors + test_errors)

    print("\n" + "=" * 60)
    print("EXPERIMENT SUMMARY")
    print("=" * 60)
    print(f"  Dataset:                 {DATASET_DIR}")
    print(f"  Classes:                 {class_names}")
    print(f"  Total images:            {num_images}")
    print(f"  Image dimensions:        {m} x {n}")
    print(f"  SVD rank k:              {SVD_RANK_K}")
    print(f"  Training images:         {len(train_images)}")
    print(f"  Testing images:          {len(test_images)}")
    print(f"  Classifier:              SVM ({CLASSIFIER_KERNEL.upper()} kernel, StandardScaler)")
    print(f"  Random seed:             {RANDOM_SEED}")
    print(f"  ---")
    print(f"  Original accuracy:       {acc_original * 100:.2f}%")
    print(f"  Compressed accuracy:     {acc_compressed * 100:.2f}%")
    print(f"  Accuracy difference:     {acc_diff:+.2f} percentage points")
    print(f"  ---")
    print(f"  Avg reconstruction error:{avg_all_error:.2f} (Frobenius norm)")
    print(f"  Avg energy retained:     {avg_energy:.2f}%")
    print(f"  Theoretical comp. ratio: {compression_ratio:.2f}x")
    print(f"  Original storage:        {m*n:,} values per image")
    print(f"  Compressed storage:      {SVD_RANK_K*(m+n+1):,} values per image")
    print(f"  ---")
    print(f"  Results saved to:        {os.path.abspath(RESULTS_DIR)}")
    print("=" * 60)
    print(f"\n  Interpretation: {interpretation}")
    print()


if __name__ == "__main__":
    main()
