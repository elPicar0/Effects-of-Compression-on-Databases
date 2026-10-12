"""
download_dataset.py — Download and Prepare Cats vs Dogs Dataset
================================================================
Downloads the dataset using `kagglehub` and populates the dataset/
folder with 350 images per class (700 images total).
"""

import os
import shutil
import cv2
import kagglehub

NUM_PER_CLASS = 350  # 350 cats + 350 dogs = 700 images total
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DST_BASE = os.path.join(PROJECT_DIR, "dataset")

print("Downloading dataset via kagglehub...")
path = kagglehub.dataset_download("bhavikjikadara/dog-and-cat-classification-dataset")
print(f"Dataset downloaded to cache: {path}")

pet_images = os.path.join(path, "PetImages")
mapping = [("Cat", "cats"), ("Dog", "dogs")]

for src_name, dst_name in mapping:
    src_dir = os.path.join(pet_images, src_name)
    dst_dir = os.path.join(DST_BASE, dst_name)
    os.makedirs(dst_dir, exist_ok=True)

    files = [f for f in os.listdir(src_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
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

    print(f"  {src_name}: copied {copied} images -> {dst_dir}")

print(f"\nDataset ready in {DST_BASE}: {NUM_PER_CLASS * 2} images total.")
