"""
preprocessing.py — Image Preprocessing Module
==============================================
Handles loading, grayscale conversion, and resizing of images using OpenCV.

Mathematical Background:
    A grayscale image is represented as a matrix A ∈ R^(m × n),
    where each element A[i, j] represents the pixel intensity at row i, column j.
    Pixel intensities range from 0 (black) to 255 (white).

    After resizing, every image becomes a matrix A ∈ R^(128 × 128).
"""

import os
import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Target image dimensions (configurable)
# ---------------------------------------------------------------------------
IMG_HEIGHT = 128
IMG_WIDTH = 128


def load_and_preprocess_image(image_path):
    """
    Load a single image and preprocess it:
        1. Read the image using cv2.imread()
        2. Convert BGR → Grayscale using cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        3. Resize to (IMG_HEIGHT × IMG_WIDTH)
        4. Return as a NumPy matrix of dtype float64

    Parameters
    ----------
    image_path : str
        Absolute or relative path to the image file.

    Returns
    -------
    gray_resized : np.ndarray, shape (IMG_HEIGHT, IMG_WIDTH)
        The preprocessed grayscale image as a matrix A ∈ R^(128 × 128).
        Values are floats in [0, 255].
    """
    # Step 1: Read the image (OpenCV loads images in BGR format by default)
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    # Step 2: Convert from BGR colour space to Grayscale
    # This maps each pixel from a 3-channel vector (B, G, R) to a single
    # intensity value using the luminosity formula:
    #   Gray = 0.299·R + 0.587·G + 0.114·B
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Step 3: Resize to a fixed dimension so every image has the same shape.
    # This ensures all feature vectors have the same length.
    gray_resized = cv2.resize(gray, (IMG_WIDTH, IMG_HEIGHT),
                              interpolation=cv2.INTER_AREA)

    # Step 4: Convert to float64 NumPy matrix for mathematical operations
    gray_resized = gray_resized.astype(np.float64)

    return gray_resized


def load_dataset(dataset_dir, max_images_per_class=None):
    """
    Load all images from a directory with the structure:
        dataset_dir/
            class_1/
                img1.jpg, img2.jpg, ...
            class_2/
                img1.jpg, img2.jpg, ...

    Each subdirectory name becomes a class label.

    Parameters
    ----------
    dataset_dir : str
        Path to the root dataset directory.
    max_images_per_class : int, optional
        Maximum number of images to load per class (default: None, loads all).

    Returns
    -------
    images : list of np.ndarray
        Each element is a (IMG_HEIGHT × IMG_WIDTH) grayscale image matrix.
    labels : list of int
        Integer labels (0, 1, ...) corresponding to each image.
    class_names : list of str
        Mapping from integer label to class name.
    """
    images = []
    labels = []

    # Get sorted class folder names for reproducibility
    class_names = sorted([
        d for d in os.listdir(dataset_dir)
        if os.path.isdir(os.path.join(dataset_dir, d))
    ])

    if len(class_names) < 2:
        raise ValueError(
            f"Expected at least 2 class folders in '{dataset_dir}', "
            f"found: {class_names}"
        )

    print(f"Found {len(class_names)} classes: {class_names}")

    for label_idx, class_name in enumerate(class_names):
        class_dir = os.path.join(dataset_dir, class_name)
        # Collect all image files (common extensions)
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        image_files = sorted([
            f for f in os.listdir(class_dir)
            if os.path.splitext(f)[1].lower() in valid_extensions
        ])

        if max_images_per_class is not None:
            image_files = image_files[:max_images_per_class]

        print(f"  Class '{class_name}' (label={label_idx}): "
              f"{len(image_files)} images")

        for img_file in image_files:
            img_path = os.path.join(class_dir, img_file)
            try:
                img_matrix = load_and_preprocess_image(img_path)
                images.append(img_matrix)
                labels.append(label_idx)
            except Exception as e:
                print(f"    Warning: Skipping {img_file} — {e}")

    return images, labels, class_names


def flatten_images(images):
    """
    Flatten each (m × n) image matrix into a 1-D feature vector.

    For a 128 × 128 image, each feature vector has 16,384 elements.

    Mathematically:
        A ∈ R^(m × n)  →  vec(A) ∈ R^(m·n)

    Parameters
    ----------
    images : list of np.ndarray
        List of 2-D image matrices.

    Returns
    -------
    X : np.ndarray, shape (num_images, m*n)
        Feature matrix where each row is a flattened image.
    """
    return np.array([img.flatten() for img in images])
