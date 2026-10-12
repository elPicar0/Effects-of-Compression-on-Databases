"""Copy 350 images per class (700 images total) from the Kaggle dataset into the project dataset folder."""
import os
import shutil
import cv2

USER_HOME = os.path.expanduser("~")
SRC_BASE = os.path.join(
    USER_HOME, ".cache", "kagglehub", "datasets", "bhavikjikadara",
    "dog-and-cat-classification-dataset", "versions", "1", "PetImages"
)
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DST_BASE = os.path.join(PROJECT_DIR, "dataset")
NUM_PER_CLASS = 350  # 350 per class = 700 images total

mapping = [("Cat", "cats"), ("Dog", "dogs")]

for src_name, dst_name in mapping:
    src_dir = os.path.join(SRC_BASE, src_name)
    dst_dir = os.path.join(DST_BASE, dst_name)
    os.makedirs(dst_dir, exist_ok=True)

    # Sort numerically for reproducibility
    files = os.listdir(src_dir)
    files = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    files.sort(key=lambda x: int(os.path.splitext(x)[0]) if os.path.splitext(x)[0].isdigit() else 99999)

    copied = 0
    for f in files:
        if copied >= NUM_PER_CLASS:
            break
        src_path = os.path.join(src_dir, f)
        img = cv2.imread(src_path)
        if img is not None:
            shutil.copy2(src_path, os.path.join(dst_dir, f))
            copied += 1

    print(f"{src_name}: copied {copied} valid images -> {dst_dir}")
