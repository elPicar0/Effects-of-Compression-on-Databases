# SVD-Based Image Compression & Classification

**3rd-Semester Mathematical Foundations of AI & Data Science — Mini-Project**

## Objective

Experimentally study how **Singular Value Decomposition (SVD)** compression affects image classification accuracy.

We compare two identical pipelines:

| Experiment | Pipeline |
|---|---|
| **Experiment A — Original** | image → grayscale → resize (128×128) → flatten → SVM |
| **Experiment B — SVD Compressed** | image → grayscale → resize (128×128) → SVD (rank $k$) → reconstruct → flatten → SVM |

Both pipelines use the **exact same**:
- Classifier: Support Vector Machine with `StandardScaler`
- Preprocessing: OpenCV Grayscale conversion and resize to $128 \times 128$
- Data split: Stratified 75% train / 25% test with fixed random seed
- Feature representation: Flattened pixel intensity vectors ($16,384$ features)

The **only** difference is whether truncated SVD compression is applied.

---

## Dataset & Setup Instructions

The dataset is **excluded from git** (`.gitignore`) to keep the repository lightweight. You can set it up in two easy ways:

### Option 1: Automatic Download via Script (Recommended)
Run the automated downloader script which fetches the dataset via `kagglehub` and prepares the 700-image balanced subset:

```bash
python download_dataset.py
```

### Option 2: Manual Download
1. Download the **Dog and Cat Classification Dataset** from Kaggle:
   🔗 [Kaggle Dataset: bhavikjikadara/dog-and-cat-classification-dataset](https://www.kaggle.com/datasets/bhavikjikadara/dog-and-cat-classification-dataset)
2. Extract the archive.
3. Place images into the following directory structure inside the project root:
   ```text
   dataset/
   ├── cats/    # ~350 images (.jpg, .jpeg, or .png)
   └── dogs/    # ~350 images (.jpg, .jpeg, or .png)
   ```

*(Note: The project also works with any binary image classification dataset placed in `dataset/<class_1>/` and `dataset/<class_2>/` folders).*

---

## Mathematical Background

### 1. Grayscale Image Representation as a Matrix

A grayscale image is represented mathematically as a real matrix:

$$A \in \mathbb{R}^{m \times n}$$

where $m = 128$ (height), $n = 128$ (width). Each matrix entry $A[i, j] \in [0, 255]$ represents the pixel intensity at row $i$ and column $j$.

### 2. Singular Value Decomposition (SVD)

Any matrix $A \in \mathbb{R}^{m \times n}$ can be factored as:

$$A = U \Sigma V^T$$

where:
- $U \in \mathbb{R}^{m \times m}$ is an orthogonal matrix of left singular vectors
- $\Sigma \in \mathbb{R}^{m \times n}$ is a diagonal matrix containing singular values $\sigma_1 \ge \sigma_2 \ge \dots \ge \sigma_r \ge 0$
- $V \in \mathbb{R}^{n \times n}$ is an orthogonal matrix of right singular vectors

### 3. Truncated SVD (Rank-$k$ Approximation)

To compress the image, we retain only the top $k$ singular values:

$$A_k = U_k \Sigma_k V_k^T = \sum_{i=1}^{k} \sigma_i \mathbf{u}_i \mathbf{v}_i^T$$

By the **Eckart–Young–Mirsky Theorem**, $A_k$ is the optimal rank-$k$ approximation of $A$ under both the Frobenius norm and the spectral norm.

### 4. SVD Measurements & Metrics

1. **Reconstruction Error (Frobenius Norm):**
   $$\|A - A_k\|_F = \sqrt{\sum_{i=1}^m \sum_{j=1}^n (A_{ij} - (A_k)_{ij})^2} = \sqrt{\sum_{i=k+1}^r \sigma_i^2}$$

2. **Energy Retained (%):**
   $$\text{Energy Retained} = \frac{\sum_{i=1}^k \sigma_i^2}{\sum_{i=1}^r \sigma_i^2} \times 100\%$$

3. **Theoretical Compression Ratio:**
   $$\text{Compression Ratio} = \frac{m \cdot n}{k \cdot (m + n + 1)}$$
   *Note: This is a theoretical scalar storage ratio, not an encoded PNG/JPEG file-size ratio.*

---

## Experimental Findings (700 Images, Ranks $k=20, 30, 50$)

| Pipeline | Accuracy (%) | Accuracy Diff vs Original | Avg Frobenius Error | Energy Retained (%) | Theoretical Compression | Storage per Image |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Original Baseline** | **53.14%** | — | 0.00 | 100.00% | 1.00x | 16,384 values |
| **SVD ($k=20$)** | **50.86%** | -2.29 pp | 992.70 | 99.57% | 3.19x | 5,140 values |
| **SVD ($k=30$)** | **52.00%** | -1.14 pp | 679.94 | 99.79% | 2.13x | 7,710 values |
| **SVD ($k=50$)** | **52.00%** | -1.14 pp | 355.16 | 99.94% | 1.28x | 12,850 values |

### Interpretation
- **Rank $k = 20$**: Saves $\approx 68.6\%$ of storage values (3.19x compression) and retains $99.57\%$ energy, with a minor accuracy drop of $2.29\text{ percentage points}$ ($53.14\% \rightarrow 50.86\%$).
- **Rank $k = 30$**: Halves the Frobenius error ($992.70 \rightarrow 679.94$), improves energy retention to $99.79\%$, and accuracy recovers to $52.00\%$ (only $-1.14\text{ pp}$ from original) with 2.13x compression.
- **Rank $k = 50$**: Reconstructs nearly the exact original image ($99.94\%$ energy, low error of $355.16$), matching the $52.00\%$ accuracy, but yields diminished compression benefits ($1.28\text{x}$).

### Rank Analysis Summary Across All Images
| Rank $k$ | Avg Frobenius Error | Energy Retained (%) | Theoretical Compression Ratio |
|:---:|:---:|:---:|:---:|
| 5 | 2487.60 | 97.44% | 12.75x |
| 10 | 1650.76 | 98.85% | 6.38x |
| 20 | 992.70 | 99.57% | 3.19x |
| 30 | 679.94 | 99.79% | 2.13x |
| 50 | 355.16 | 99.94% | 1.28x |

---

## Project Structure

```
svd_image_classification/
│
├── dataset/
│   ├── cats/                          # 350 cat images
│   └── dogs/                          # 350 dog images
│
├── results/
│   ├── accuracy_comparison.png        # Bar chart: Original vs SVD accuracy
│   ├── error_vs_rank.png              # Frobenius reconstruction error vs k
│   ├── energy_vs_rank.png             # Energy retained (%) vs k
│   ├── compression_vs_rank.png        # Theoretical compression ratio vs k
│   ├── visual_original.png            # Original test image
│   ├── visual_rank_5.png              # SVD reconstruction at k=5
│   ├── visual_rank_10.png             # SVD reconstruction at k=10
│   ├── visual_rank_20.png             # SVD reconstruction at k=20
│   ├── visual_rank_30.png             # SVD reconstruction at k=30
│   ├── visual_rank_50.png             # SVD reconstruction at k=50
│   └── visual_comparison_grid.png     # Side-by-side reconstruction grid
│
├── src/
│   ├── __init__.py
│   ├── preprocessing.py               # cv2 grayscale, 128x128 resize, matrix conversion
│   ├── svd_compression.py             # SVD decomposition, rank-k truncation, metrics
│   ├── classifier.py                  # Support Vector Machine with StandardScaler
│   └── experiment.py                  # Plotting routines and visual comparisons
│
├── main.py                            # Main experiment pipeline runner
├── download_dataset.py                # Automated Kaggle dataset downloader
├── copy_dataset.py                    # Helper to populate dataset from cache
├── requirements.txt                   # Project dependencies
└── README.md                          # Project documentation
```

---

## Setup and Execution

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Experiment
```bash
python main.py
```

### 3. Configurable Parameters (in `main.py`)
At the top of [`main.py`](file:///c:/Users/shaun/Desktop/SVD/svd_image_classification/main.py):
```python
SVD_RANK_K = 20                        # Target SVD rank
ANALYSIS_RANKS = [5, 10, 20, 30, 50]   # Ranks for analysis curves
MAX_IMAGES_PER_CLASS = 350             # 350 per class = 700 images total
CLASSIFIER_KERNEL = 'rbf'              # 'rbf' or 'linear'
TEST_SIZE = 0.25                       # Train/test split ratio (25% test)
RANDOM_SEED = 42                       # Seed for reproducibility
```
