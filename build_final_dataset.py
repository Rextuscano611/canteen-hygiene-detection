"""
Build Final Merged Training Dataset - Canteen Hygiene Project
---------------------------------------------------------------
Combines:
  1. A curated random sample from the OLD dataset (general prior)
  2. Your ORIGINAL 376 CCTV images (real domain signal)
  3. AUGMENTED copies of CCTV images that contain no_hair_cap
     (your rare/important class, multiplied 6-8x)

Final train/valid/test split:
  - train: curated old sample + CCTV train + augmented no_hair_cap CCTV
  - valid: CCTV valid images ONLY (clean, real-domain validation)
  - test:  CCTV test images ONLY (clean, real-domain test)

Usage:
    pip install albumentations opencv-python
    python build_final_dataset.py
"""

import os
import random
import shutil

import cv2

try:
    import albumentations as A
except ImportError:
    raise SystemExit("Please run: pip install albumentations opencv-python")


# ============================================================
# CONFIG — all paths resolve relative to this script's location
# ============================================================
BASE = os.path.dirname(os.path.abspath(__file__))

OLD_DATASET  = os.path.join(BASE, "final_dem_2.v1-m_n.yolo26")
CCTV_DATASET = os.path.join(BASE, "canteen-hygiene-cctv.v1-version_1_hair_cap.yolo26")
FINAL_DIR    = os.path.join(BASE, "final_merged_dataset")

OLD_TRAIN_SAMPLE_SIZE   = 900
NOCAP_AUGMENT_MULTIPLIER = 7
RANDOM_SEED             = 42
NOCAP_CLASS_ID          = "1"

random.seed(RANDOM_SEED)


transform = A.Compose(
    [
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=12, p=0.5, border_mode=cv2.BORDER_CONSTANT),
        A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=0.6),
        A.RandomResizedCrop(size=(1280, 1280), scale=(0.85, 1.0), p=0.4),
        A.GaussNoise(p=0.15),
        A.MotionBlur(blur_limit=5, p=0.15),
    ],
    bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"], min_visibility=0.3),
)


def read_yolo_labels(label_path):
    """Read a YOLO .txt label file -> (bboxes, class_labels).

    Some exported labels (e.g. from SAM3 mask->box conversion near
    image edges) have x_center/width combinations whose actual edges
    (x_center +/- width/2) fall outside [0,1], even though the raw
    numbers look individually plausible. We fix this by converting to
    absolute edges, clamping THOSE to [0,1], then converting back to
    YOLO center/width format.
    """
    bboxes, class_labels = [], []
    if not os.path.exists(label_path):
        return bboxes, class_labels
    with open(label_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            cls = int(parts[0])
            x, y, w, h = map(float, parts[1:5])

            # convert to absolute edges
            x_min = x - w / 2
            x_max = x + w / 2
            y_min = y - h / 2
            y_max = y + h / 2

            # clamp edges to valid [0,1] range
            x_min = min(max(x_min, 0.0), 1.0)
            x_max = min(max(x_max, 0.0), 1.0)
            y_min = min(max(y_min, 0.0), 1.0)
            y_max = min(max(y_max, 0.0), 1.0)

            # skip degenerate boxes (zero or negative area after clamping)
            if x_max <= x_min or y_max <= y_min:
                continue

            # convert back to YOLO center/width format
            new_w = x_max - x_min
            new_h = y_max - y_min
            new_x = x_min + new_w / 2
            new_y = y_min + new_h / 2

            # final safety clip (floating point edge cases)
            new_x = min(max(new_x, 1e-6), 1.0 - 1e-6)
            new_y = min(max(new_y, 1e-6), 1.0 - 1e-6)
            new_w = min(max(new_w, 1e-6), 1.0)
            new_h = min(max(new_h, 1e-6), 1.0)

            bboxes.append([new_x, new_y, new_w, new_h])
            class_labels.append(cls)
    return bboxes, class_labels


def write_yolo_labels(label_path, bboxes, class_labels):
    with open(label_path, "w") as f:
        for cls, box in zip(class_labels, bboxes):
            f.write(f"{cls} {' '.join(f'{v:.6f}' for v in box)}\n")


def label_has_class(label_path, class_id):
    if not os.path.exists(label_path):
        return False
    with open(label_path, "r") as f:
        for line in f:
            if line.strip().split(" ")[0] == class_id:
                return True
    return False


def ensure_dirs():
    for split in ["train", "valid", "test"]:
        os.makedirs(os.path.join(FINAL_DIR, split, "images"), exist_ok=True)
        os.makedirs(os.path.join(FINAL_DIR, split, "labels"), exist_ok=True)


def copy_pair(img_path, label_path, out_img_dir, out_label_dir, new_name):
    ext = os.path.splitext(img_path)[1]
    shutil.copy(img_path, os.path.join(out_img_dir, f"{new_name}{ext}"))
    if os.path.exists(label_path):
        shutil.copy(label_path, os.path.join(out_label_dir, f"{new_name}.txt"))
    else:
        open(os.path.join(out_label_dir, f"{new_name}.txt"), "w").close()


def step1_sample_old_dataset():
    print("=" * 60)
    print("STEP 1: Sampling curated subset from OLD dataset")
    print("=" * 60)

    old_img_dir = os.path.join(OLD_DATASET, "train", "images")
    old_lbl_dir = os.path.join(OLD_DATASET, "train", "labels")

    all_images = [f for f in os.listdir(old_img_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    print(f"  Old dataset train images available: {len(all_images)}")

    sample_size = min(OLD_TRAIN_SAMPLE_SIZE, len(all_images))
    sampled = random.sample(all_images, sample_size)

    out_img_dir = os.path.join(FINAL_DIR, "train", "images")
    out_lbl_dir = os.path.join(FINAL_DIR, "train", "labels")

    for i, img_file in enumerate(sampled):
        base = os.path.splitext(img_file)[0]
        img_path = os.path.join(old_img_dir, img_file)
        label_path = os.path.join(old_lbl_dir, f"{base}.txt")
        copy_pair(img_path, label_path, out_img_dir, out_lbl_dir, f"old_{i:04d}")

    print(f"  -> Copied {len(sampled)} images from old dataset into final train set\n")
    return len(sampled)


def step2_copy_cctv_train():
    print("=" * 60)
    print("STEP 2: Copying original CCTV train images")
    print("=" * 60)

    cctv_img_dir = os.path.join(CCTV_DATASET, "train", "images")
    cctv_lbl_dir = os.path.join(CCTV_DATASET, "train", "labels")

    all_images = [f for f in os.listdir(cctv_img_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    print(f"  CCTV train images found: {len(all_images)}")

    out_img_dir = os.path.join(FINAL_DIR, "train", "images")
    out_lbl_dir = os.path.join(FINAL_DIR, "train", "labels")

    nocap_images = []

    for i, img_file in enumerate(all_images):
        base = os.path.splitext(img_file)[0]
        img_path = os.path.join(cctv_img_dir, img_file)
        label_path = os.path.join(cctv_lbl_dir, f"{base}.txt")
        new_name = f"cctv_{i:04d}"
        copy_pair(img_path, label_path, out_img_dir, out_lbl_dir, new_name)

        if label_has_class(label_path, NOCAP_CLASS_ID):
            nocap_images.append((img_path, label_path, new_name))

    print(f"  -> Copied {len(all_images)} CCTV images into final train set")
    print(f"  -> Found {len(nocap_images)} images containing no_hair_cap\n")
    return nocap_images


def step3_augment_nocap(nocap_images):
    print("=" * 60)
    print(f"STEP 3: Augmenting no_hair_cap images ({NOCAP_AUGMENT_MULTIPLIER}x)")
    print("=" * 60)

    out_img_dir = os.path.join(FINAL_DIR, "train", "images")
    out_lbl_dir = os.path.join(FINAL_DIR, "train", "labels")

    total_augmented = 0
    for img_path, label_path, base_name in nocap_images:
        image = cv2.imread(img_path)
        if image is None:
            print(f"  [WARN] Could not read image: {img_path}")
            continue

        bboxes, class_labels = read_yolo_labels(label_path)
        if not bboxes:
            continue

        for aug_idx in range(NOCAP_AUGMENT_MULTIPLIER):
            try:
                augmented = transform(image=image, bboxes=bboxes, class_labels=class_labels)
            except Exception as e:
                print(f"  [WARN] Augmentation failed on {base_name}_aug{aug_idx}: {e}")
                continue

            if not augmented["bboxes"]:
                continue

            out_name = f"{base_name}_aug{aug_idx}"
            out_img_path = os.path.join(out_img_dir, f"{out_name}.jpg")
            out_lbl_path = os.path.join(out_lbl_dir, f"{out_name}.txt")

            cv2.imwrite(out_img_path, augmented["image"])
            write_yolo_labels(out_lbl_path, augmented["bboxes"], augmented["class_labels"])
            total_augmented += 1

    print(f"  -> Generated {total_augmented} augmented no_hair_cap images\n")
    return total_augmented


def step4_copy_valid_test():
    print("=" * 60)
    print("STEP 4: Copying CCTV valid/test sets (clean, unaugmented)")
    print("=" * 60)

    for split in ["valid", "test"]:
        src_img_dir = os.path.join(CCTV_DATASET, split, "images")
        src_lbl_dir = os.path.join(CCTV_DATASET, split, "labels")
        out_img_dir = os.path.join(FINAL_DIR, split, "images")
        out_lbl_dir = os.path.join(FINAL_DIR, split, "labels")

        images = [f for f in os.listdir(src_img_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        for i, img_file in enumerate(images):
            base = os.path.splitext(img_file)[0]
            img_path = os.path.join(src_img_dir, img_file)
            label_path = os.path.join(src_lbl_dir, f"{base}.txt")
            copy_pair(img_path, label_path, out_img_dir, out_lbl_dir, f"cctv_{split}_{i:04d}")

        print(f"  -> {split}: {len(images)} images copied (CCTV only, no old-dataset data)")

    print()


def write_data_yaml():
    yaml_path = os.path.join(FINAL_DIR, "data.yaml")
    # Use a relative path marker so the yaml works on any machine
    content = """path: .
train: train/images
val:   valid/images
test:  test/images

nc: 2
names: ['hair_cap', 'no_hair_cap']
"""
    with open(yaml_path, "w") as f:
        f.write(content)
    print(f"Wrote data.yaml -> {yaml_path}\n")


def main():
    if os.path.exists(FINAL_DIR):
        print(f"Removing existing folder: {FINAL_DIR}")
        shutil.rmtree(FINAL_DIR)
    ensure_dirs()

    old_count    = step1_sample_old_dataset()
    nocap_images = step2_copy_cctv_train()
    aug_count    = step3_augment_nocap(nocap_images)
    step4_copy_valid_test()
    write_data_yaml()

    final_train_count = len(os.listdir(os.path.join(FINAL_DIR, "train", "images")))
    final_valid_count = len(os.listdir(os.path.join(FINAL_DIR, "valid", "images")))
    final_test_count  = len(os.listdir(os.path.join(FINAL_DIR, "test",  "images")))

    print("=" * 60)
    print("DONE - FINAL MERGED DATASET SUMMARY")
    print("=" * 60)
    print(f"  Old dataset (curated sample):    {old_count}")
    print(f"  CCTV images with no_hair_cap:    {len(nocap_images)}")
    print(f"  Augmented no_hair_cap images:    {aug_count}")
    print(f"  TOTAL train images:              {final_train_count}")
    print(f"  TOTAL valid images (CCTV only):  {final_valid_count}")
    print(f"  TOTAL test  images (CCTV only):  {final_test_count}")
    print("=" * 60)


if __name__ == "__main__":
    main()
