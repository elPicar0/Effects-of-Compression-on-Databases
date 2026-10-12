"""
svd_compression.py — SVD Image Compression Module
===================================================
Implements Singular Value Decomposition (SVD) for image compression.

Mathematical Background:
    For any real matrix A ∈ R^(m × n), the SVD factorisation is:

        A = U Σ Vᵀ

    where:
        U ∈ R^(m × m)   — left singular vectors  (orthogonal matrix)
        Σ ∈ R^(m × n)   — diagonal matrix of singular values σ₁ ≥ σ₂ ≥ ... ≥ 0
        V ∈ R^(n × n)   — right singular vectors (orthogonal matrix)

    Truncated (rank-k) approximation:
        A_k = U_k Σ_k V_kᵀ

    where only the first k singular values/vectors are retained.
    By the Eckart–Young theorem, A_k is the best rank-k approximation
    of A in both the Frobenius and spectral norms.

    Storage comparison:
        Original:  m × n  values
        Truncated: k × (m + n + 1) values   [U_k, Σ_k, V_kᵀ]
"""

import numpy as np


def svd_compress(image_matrix, k):
    """
    Compress a grayscale image using truncated SVD.

    Parameters
    ----------
    image_matrix : np.ndarray, shape (m, n)
        The original grayscale image matrix A ∈ R^(m × n).
    k : int
        Number of singular values to retain (rank of approximation).
        Must satisfy 1 ≤ k ≤ min(m, n).

    Returns
    -------
    reconstructed : np.ndarray, shape (m, n)
        The rank-k approximation A_k, with pixel values clipped to [0, 255].
    U_k : np.ndarray, shape (m, k)
        First k left singular vectors.
    sigma_k : np.ndarray, shape (k,)
        First k singular values.
    Vt_k : np.ndarray, shape (k, n)
        First k right singular vectors (transposed).
    all_sigmas : np.ndarray
        All singular values of the original matrix (for analysis).
    """
    m, n = image_matrix.shape

    # Clamp k to the maximum possible rank
    max_rank = min(m, n)
    if k > max_rank:
        print(f"  Warning: k={k} exceeds max rank={max_rank}. Using k={max_rank}.")
        k = max_rank

    # -----------------------------------------------------------------------
    # Step 1: Compute the full SVD of the image matrix
    #     A = U Σ Vᵀ
    #
    # np.linalg.svd returns:
    #   U  — shape (m, m)  if full_matrices=True, or (m, min(m,n)) if False
    #   s  — shape (min(m, n),)  — the singular values as a 1-D array
    #   Vt — shape (n, n)  if full_matrices=True, or (min(m,n), n) if False
    # -----------------------------------------------------------------------
    U, s, Vt = np.linalg.svd(image_matrix, full_matrices=False)

    # -----------------------------------------------------------------------
    # Step 2: Truncate to rank k
    #     Keep only the first k columns of U, first k singular values,
    #     and first k rows of Vᵀ.
    # -----------------------------------------------------------------------
    U_k = U[:, :k]       # shape: (m, k)
    sigma_k = s[:k]      # shape: (k,)
    Vt_k = Vt[:k, :]     # shape: (k, n)

    # -----------------------------------------------------------------------
    # Step 3: Reconstruct the approximated image
    #     A_k = U_k · diag(σ_k) · V_kᵀ
    #
    # We multiply U_k by diag(sigma_k) first, then by Vt_k.
    # Equivalent to: A_k = (U_k * sigma_k) @ Vt_k   (broadcasting)
    # -----------------------------------------------------------------------
    reconstructed = (U_k * sigma_k) @ Vt_k  # shape: (m, n)

    # -----------------------------------------------------------------------
    # Step 4: Clip pixel values to valid range [0, 255]
    #     The SVD reconstruction may produce values slightly outside this
    #     range due to floating-point arithmetic.
    # -----------------------------------------------------------------------
    reconstructed = np.clip(reconstructed, 0, 255)

    return reconstructed, U_k, sigma_k, Vt_k, s


def compute_reconstruction_error(original, reconstructed):
    """
    Compute the reconstruction error using the Frobenius norm.

    The Frobenius norm of a matrix A is:
        ||A||_F = sqrt( Σᵢ Σⱼ |aᵢⱼ|² )

    The reconstruction error is:
        ||A - A_k||_F

    This measures how much information is lost in the compression.

    Parameters
    ----------
    original : np.ndarray, shape (m, n)
    reconstructed : np.ndarray, shape (m, n)

    Returns
    -------
    error : float
        The Frobenius norm of the difference matrix.
    """
    diff = original - reconstructed
    error = np.linalg.norm(diff, 'fro')
    return error


def compute_energy_retained(all_sigmas, k):
    """
    Compute the fraction of "energy" retained by keeping k singular values.

    Energy retained (%) = ( Σᵢ₌₁ᵏ σᵢ² ) / ( Σᵢ₌₁ʳ σᵢ² ) × 100

    where r = min(m, n) is the total number of singular values.

    This metric tells us what proportion of the total variance in the
    image is captured by the rank-k approximation.

    Parameters
    ----------
    all_sigmas : np.ndarray
        All singular values of the original matrix.
    k : int
        Number of singular values retained.

    Returns
    -------
    energy_pct : float
        Energy retained as a percentage (0–100).
    """
    # Sum of squares of all singular values (total energy)
    total_energy = np.sum(all_sigmas ** 2)

    # Sum of squares of the first k singular values (retained energy)
    retained_energy = np.sum(all_sigmas[:k] ** 2)

    if total_energy == 0:
        return 0.0

    energy_pct = (retained_energy / total_energy) * 100.0
    return energy_pct


def compute_compression_ratio(m, n, k):
    """
    Compute the theoretical compression ratio.

    Original storage:   m × n  values
    Truncated storage:  k × (m + n + 1)  values
        — k columns of U  → m × k
        — k singular values → k
        — k rows of Vᵀ    → k × n

    Compression Ratio = (m × n) / [k × (m + n + 1)]

    A ratio > 1 means compression (fewer values stored).
    A ratio < 1 means expansion  (more values stored than original).

    NOTE: This is a *theoretical* compression ratio based on the number
    of scalar values stored, NOT an actual file-size ratio.

    Parameters
    ----------
    m, n : int
        Dimensions of the original image matrix.
    k : int
        Number of singular values retained.

    Returns
    -------
    ratio : float
        The theoretical compression ratio.
    """
    original_storage = m * n
    truncated_storage = k * (m + n + 1)
    ratio = original_storage / truncated_storage
    return ratio


def compress_image_batch(images, k):
    """
    Apply SVD compression to a batch of images.

    Parameters
    ----------
    images : list of np.ndarray
        List of grayscale image matrices.
    k : int
        SVD rank for compression.

    Returns
    -------
    compressed_images : list of np.ndarray
        List of reconstructed (compressed) image matrices.
    errors : list of float
        Frobenius reconstruction error for each image.
    energies : list of float
        Energy retained (%) for each image.
    """
    compressed_images = []
    errors = []
    energies = []

    for img in images:
        reconstructed, _, _, _, all_sigmas = svd_compress(img, k)
        compressed_images.append(reconstructed)
        errors.append(compute_reconstruction_error(img, reconstructed))
        energies.append(compute_energy_retained(all_sigmas, k))

    return compressed_images, errors, energies
