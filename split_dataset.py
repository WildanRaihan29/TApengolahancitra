import os
import shutil
from sklearn.model_selection import train_test_split

SOURCE_DIR = "dataset"

TRAIN_DIR = "dataset_split/train"
VAL_DIR = "dataset_split/validation"
TEST_DIR = "dataset_split/test"

classes = [
    "blast",
    "blight",
    "brown_spot",
    "hispa",
    "normal",
    "tungro"
]

for cls in classes:

    source_folder = os.path.join(SOURCE_DIR, cls)

    if not os.path.exists(source_folder):
        print(f"Folder tidak ditemukan: {source_folder}")
        continue

    os.makedirs(os.path.join(TRAIN_DIR, cls), exist_ok=True)
    os.makedirs(os.path.join(VAL_DIR, cls), exist_ok=True)
    os.makedirs(os.path.join(TEST_DIR, cls), exist_ok=True)

    images = [
        f for f in os.listdir(source_folder)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    train_imgs, temp_imgs = train_test_split(
        images,
        test_size=0.30,
        random_state=42
    )

    val_imgs, test_imgs = train_test_split(
        temp_imgs,
        test_size=0.50,
        random_state=42
    )

    for img in train_imgs:
        shutil.copy(
            os.path.join(source_folder, img),
            os.path.join(TRAIN_DIR, cls, img)
        )

    for img in val_imgs:
        shutil.copy(
            os.path.join(source_folder, img),
            os.path.join(VAL_DIR, cls, img)
        )

    for img in test_imgs:
        shutil.copy(
            os.path.join(source_folder, img),
            os.path.join(TEST_DIR, cls, img)
        )

    print(
        f"{cls}: "
        f"train={len(train_imgs)}, "
        f"val={len(val_imgs)}, "
        f"test={len(test_imgs)}"
    )

print("\nDataset berhasil dibagi!")