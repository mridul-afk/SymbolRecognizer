import os
import cv2
from tqdm import tqdm

DATASET_DIR = "dataset"
IMG_SIZE = 128

for split in ["train", "validation", "test"]:
    split_path = os.path.join(DATASET_DIR, split)

    for cls in os.listdir(split_path):
        class_path = os.path.join(split_path, cls)

        for img_name in tqdm(os.listdir(class_path), desc=f"{split}-{cls}"):
            img_path = os.path.join(class_path, img_name)

            image = cv2.imread(img_path)

            if image is None:
                continue

            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Resize
            resized = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))

            # Overwrite image
            cv2.imwrite(img_path, resized)

print("All images resized to 128x128 and converted to grayscale!")
