"""
experiment.py — Experiment Runner & Visualisation Module
=========================================================
Runs both classification experiments and generates all plots and outputs.
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend (no GUI needed)
import matplotlib.pyplot as plt

from src.svd_compression import (
    svd_compress,
    compress_image_batch,
    compute_reconstruction_error,
    compute_energy_retained,
    compute_compression_ratio,
)


def run_rank_analysis(images, ranks, results_dir):
    """
    For each rank in `ranks`, compute average reconstruction error,
    energy retained, and compression ratio across all images.
    Generate and save the corresponding plots.

    Parameters
    ----------
    images : list of np.ndarray
        Grayscale image matrices (from the full dataset).
    ranks : list of int
        List of SVD ranks to analyse.
    results_dir : str
        Directory to save plots.
    """
    m, n = images[0].shape
    avg_errors = []
    avg_energies = []
    ratios = []

    print("\n--- Rank Analysis ---")
    print(f"{'Rank k':>8}  {'Avg Error':>12}  {'Energy (%)':>12}  {'Comp. Ratio':>12}")
    print("-" * 52)

    for k in ranks:
        _, errors, energies = compress_image_batch(images, k)
        avg_err = np.mean(errors)
        avg_eng = np.mean(energies)
        ratio = compute_compression_ratio(m, n, k)

        avg_errors.append(avg_err)
        avg_energies.append(avg_eng)
        ratios.append(ratio)

        print(f"{k:>8d}  {avg_err:>12.2f}  {avg_eng:>11.2f}%  {ratio:>12.2f}x")

    # --- Plot 1: Rank vs Reconstruction Error ---
    plt.figure(figsize=(8, 5))
    plt.plot(ranks, avg_errors, 'bo-', linewidth=2, markersize=8)
    plt.xlabel('Rank k (number of singular values retained)', fontsize=12)
    plt.ylabel('Average Frobenius Reconstruction Error', fontsize=12)
    plt.title('Rank vs Reconstruction Error', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'error_vs_rank.png'), dpi=150)
    plt.close()
    print(f"\nSaved: {os.path.join(results_dir, 'error_vs_rank.png')}")

    # --- Plot 2: Rank vs Energy Retained ---
    plt.figure(figsize=(8, 5))
    plt.plot(ranks, avg_energies, 'gs-', linewidth=2, markersize=8)
    plt.xlabel('Rank k (number of singular values retained)', fontsize=12)
    plt.ylabel('Average Energy Retained (%)', fontsize=12)
    plt.title('Rank vs Energy Retained', fontsize=14)
    plt.ylim(0, 105)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'energy_vs_rank.png'), dpi=150)
    plt.close()
    print(f"Saved: {os.path.join(results_dir, 'energy_vs_rank.png')}")

    # --- Plot 3: Rank vs Theoretical Compression Ratio ---
    plt.figure(figsize=(8, 5))
    plt.plot(ranks, ratios, 'r^-', linewidth=2, markersize=8)
    plt.xlabel('Rank k (number of singular values retained)', fontsize=12)
    plt.ylabel('Theoretical Compression Ratio', fontsize=12)
    plt.title('Rank vs Theoretical Compression Ratio', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'compression_vs_rank.png'), dpi=150)
    plt.close()
    print(f"Saved: {os.path.join(results_dir, 'compression_vs_rank.png')}")

    return avg_errors, avg_energies, ratios


def plot_accuracy_comparison(acc_original, acc_compressed, k, results_dir):
    """
    Create a bar chart comparing classification accuracies.

    Parameters
    ----------
    acc_original : float
        Accuracy on original images (0.0 to 1.0).
    acc_compressed : float
        Accuracy on SVD-compressed images (0.0 to 1.0).
    k : int
        SVD rank used for compression.
    results_dir : str
        Directory to save the plot.
    """
    labels = ['Original Images', f'SVD Compressed (k={k})']
    accuracies = [acc_original * 100, acc_compressed * 100]
    colors = ['#2196F3', '#FF9800']

    plt.figure(figsize=(8, 5))
    bars = plt.bar(labels, accuracies, color=colors, width=0.5, edgecolor='black')

    # Add value labels on top of each bar
    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                 f'{acc:.1f}%', ha='center', va='bottom', fontsize=14,
                 fontweight='bold')

    plt.ylabel('Classification Accuracy (%)', fontsize=12)
    plt.title('Original vs SVD-Compressed Image Classification', fontsize=14)
    plt.ylim(0, 110)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'accuracy_comparison.png'), dpi=150)
    plt.close()
    print(f"Saved: {os.path.join(results_dir, 'accuracy_comparison.png')}")


def save_visual_comparison(image, ranks, results_dir):
    """
    For a single image, save the original and SVD reconstructions at
    various ranks for visual comparison.

    Parameters
    ----------
    image : np.ndarray, shape (m, n)
        A grayscale test image.
    ranks : list of int
        Ranks to visualise (e.g. [5, 10, 20, 50]).
    results_dir : str
        Directory to save the images.
    """
    # Save original
    plt.figure(figsize=(4, 4))
    plt.imshow(image, cmap='gray', vmin=0, vmax=255)
    plt.title('Original Image', fontsize=12)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'visual_original.png'), dpi=150)
    plt.close()
    print(f"\nSaved: {os.path.join(results_dir, 'visual_original.png')}")

    # Save reconstructions for each rank
    for k in ranks:
        reconstructed, _, _, _, all_sigmas = svd_compress(image, k)
        error = compute_reconstruction_error(image, reconstructed)
        energy = compute_energy_retained(all_sigmas, k)

        plt.figure(figsize=(4, 4))
        plt.imshow(reconstructed, cmap='gray', vmin=0, vmax=255)
        plt.title(f'Rank k={k}  (Energy: {energy:.1f}%)', fontsize=11)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, f'visual_rank_{k}.png'), dpi=150)
        plt.close()
        print(f"Saved: {os.path.join(results_dir, f'visual_rank_{k}.png')}")

    # Also create a single combined figure for presentations
    fig, axes = plt.subplots(1, len(ranks) + 1, figsize=(4 * (len(ranks) + 1), 4))

    axes[0].imshow(image, cmap='gray', vmin=0, vmax=255)
    axes[0].set_title('Original', fontsize=11)
    axes[0].axis('off')

    for i, k in enumerate(ranks):
        reconstructed, _, _, _, all_sigmas = svd_compress(image, k)
        energy = compute_energy_retained(all_sigmas, k)
        axes[i + 1].imshow(reconstructed, cmap='gray', vmin=0, vmax=255)
        axes[i + 1].set_title(f'k={k} ({energy:.1f}%)', fontsize=11)
        axes[i + 1].axis('off')

    plt.suptitle('SVD Image Compression — Visual Comparison', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'visual_comparison_grid.png'), dpi=150)
    plt.close()
    print(f"Saved: {os.path.join(results_dir, 'visual_comparison_grid.png')}")
