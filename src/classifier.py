"""
classifier.py — Classification Module
=======================================
Provides a simple classifier for the image classification experiments.

We use a Support Vector Machine (SVM) with a linear kernel because:
  - It works well on high-dimensional data (flattened 128×128 = 16,384 features)
  - It is mathematically transparent (finds a separating hyperplane)
  - It does not require deep learning

The classifier is identical for both experiments (original vs SVD-compressed).
"""

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import numpy as np


def create_classifier(kernel='rbf'):
    """
    Create and return a Support Vector Machine classifier.

    Parameters
    ----------
    kernel : str, default='rbf'
        Kernel type to use in the SVM ('rbf' or 'linear').

    Mathematical intuition:
        - Linear kernel: Finds a hyperplane w·x + b = 0 in the original
          feature space R^(m·n).
        - RBF (Radial Basis Function) kernel: Maps data into an infinite-
          dimensional Hilbert space via the Gaussian kernel:
              K(x, x') = exp(-gamma * ||x - x'||^2)
          and finds a separating hyperplane there. This is well-suited
          for image pixels where class boundaries are non-linear.

    Returns
    -------
    clf : sklearn.svm.SVC
        An untrained SVM classifier.
    """
    clf = SVC(kernel=kernel, random_state=42)
    return clf


def train_and_evaluate(X_train, y_train, X_test, y_test, kernel='rbf'):
    """
    Train a classifier on the training set and evaluate on the test set.

    Pipeline:
        1. Standardise features (zero mean, unit variance) — important for SVM
        2. Train the SVM classifier on the training data
        3. Predict labels on the test data
        4. Compute and return the accuracy

    Parameters
    ----------
    X_train : np.ndarray, shape (n_train, n_features)
        Training feature vectors (flattened images).
    y_train : np.ndarray, shape (n_train,)
        Training labels.
    X_test : np.ndarray, shape (n_test, n_features)
        Test feature vectors (flattened images).
    y_test : np.ndarray, shape (n_test,)
        Test labels.
    kernel : str, default='rbf'
        SVM kernel ('rbf' or 'linear').

    Returns
    -------
    accuracy : float
        Classification accuracy on the test set (0.0 to 1.0).
    predictions : np.ndarray
        Predicted labels for the test set.
    """
    # Step 1: Feature scaling (standardisation)
    # Each feature (pixel position) is scaled to have mean=0, std=1.
    # This is important because SVM is sensitive to feature magnitudes.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)  # Use SAME scaler (no data leakage)

    # Step 2: Train the SVM classifier
    clf = create_classifier(kernel=kernel)
    clf.fit(X_train_scaled, y_train)

    # Step 3: Predict on the test set
    predictions = clf.predict(X_test_scaled)

    # Step 4: Compute accuracy
    #   Accuracy = (number of correct predictions) / (total predictions)
    accuracy = accuracy_score(y_test, predictions)

    return accuracy, predictions
