import os
import random
import shutil

# ----------------------------
# SETTINGS
# ----------------------------

SOURCE_DATASET = "reference"

OUTPUT_DATASET = "training/dataset"

TRAIN_RATIO = 0.80

VALID_EXTENSIONS = (
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tif",
    ".tiff",
)

RANDOM_SEED = 42

# ----------------------------

random.seed(RANDOM_SEED)

train_root = os.path.join(OUTPUT_DATASET, "train")
val_root = os.path.join(OUTPUT_DATASET, "val")

os.makedirs(train_root, exist_ok=True)
os.makedirs(val_root, exist_ok=True)

total_train = 0
total_val = 0

print("=" * 60)
print("Creating Train / Validation Split")
print("=" * 60)

for class_name in sorted(os.listdir(SOURCE_DATASET)):

    class_path = os.path.join(SOURCE_DATASET, class_name)

    if not os.path.isdir(class_path):
        continue

    images = []

    for file in os.listdir(class_path):

        if file.lower().endswith(VALID_EXTENSIONS):

            images.append(file)

    random.shuffle(images)

    split_index = int(len(images) * TRAIN_RATIO)

    train_images = images[:split_index]
    val_images = images[split_index:]

    train_class = os.path.join(train_root, class_name)
    val_class = os.path.join(val_root, class_name)

    os.makedirs(train_class, exist_ok=True)
    os.makedirs(val_class, exist_ok=True)

    for img in train_images:

        shutil.copy2(
            os.path.join(class_path, img),
            os.path.join(train_class, img)
        )

    for img in val_images:

        shutil.copy2(
            os.path.join(class_path, img),
            os.path.join(val_class, img)
        )

    total_train += len(train_images)
    total_val += len(val_images)

    print(
        f"{class_name:25s}"
        f" Train: {len(train_images):4d}"
        f"   Val: {len(val_images):4d}"
    )

print()

print("=" * 60)
print("Completed")
print("=" * 60)

print(f"Training Images  : {total_train}")
print(f"Validation Images: {total_val}")