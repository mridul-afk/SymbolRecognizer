import os
import cv2
from tqdm import tqdm
import albumentations as A
from sklearn.model_selection import train_test_split

RAW_DIR = "dataset_raw"
OUTPUT_DIR = "Dataset"

AUG_PER_IMAGE = 5  

transform = A.Compose([
    A.RandomBrightnessContrast(p=0.5),
    A.GaussNoise(p=0.3),
    A.Rotate(limit=15, p=0.7),
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=10, p=0.5),
    A.MotionBlur(p=0.2),
    A.RandomShadow(p=0.3),
])

def process_class(class_name):
    class_path = os.path.join(RAW_DIR, class_name)
    images = os.listdir(class_path)

    train_imgs, temp_imgs = train_test_split(images, test_size=0.3, random_state=42)
    val_imgs, test_imgs = train_test_split(temp_imgs, test_size=0.5, random_state=42)

    splits = {
        "train": train_imgs,
        "validation": val_imgs,
        "test": test_imgs
    }

    for split in splits:
        for img_name in tqdm(splits[split], desc=f"{class_name}-{split}"):

            img_path = os.path.join(class_path, img_name)
            image = cv2.imread(img_path)

            if image is None:
                continue

            base_name = img_name.split('.')[0]

            # Save original
            save_path = os.path.join(OUTPUT_DIR, split, class_name, img_name)
            cv2.imwrite(save_path, image)

            # Augment
            for i in range(AUG_PER_IMAGE):
                augmented = transform(image=image)["image"]
                aug_name = f"{base_name}_aug_{i}.jpg"
                aug_path = os.path.join(OUTPUT_DIR, split, class_name, aug_name)
                cv2.imwrite(aug_path, augmented)

def main():
    classes = os.listdir(RAW_DIR)

    for cls in classes:
        process_class(cls)

    print("Dataset transfer + augmentation complete!")

if __name__ == "__main__":
    main()
