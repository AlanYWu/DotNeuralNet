import os
import shutil
import random
import json
from glob import glob

# Parameters
YOLO_LABELS_DIR = 'yolo_labels'  # where generate_yolo_annotations.py output is
OUTPUT_DIR = 'yolo_dataset'      # where to put split data
TRAIN_RATIO = 0.8
SEED = 42

random.seed(SEED)

# Helper: find all images and their label files
def collect_image_label_pairs(root):
    pairs = []
    for subdir, _, files in os.walk(root):
        for file in files:
            if file.endswith('.jpg') or file.endswith('.jpeg') or file.endswith('.png'):
                img_path = os.path.join(subdir, file)
                label_path = os.path.splitext(img_path)[0] + '.txt'
                if os.path.exists(label_path):
                    pairs.append((img_path, label_path))
    return pairs

# Collect all pairs from both Angelina and DSBI
def collect_all_pairs():
    pairs = []
    for dataset in ['Angelina', 'DSBI']:
        root = os.path.join(YOLO_LABELS_DIR, dataset)
        if os.path.exists(root):
            pairs.extend(collect_image_label_pairs(root))
    return pairs

# Split and copy files
def split_and_copy(pairs, output_dir, train_ratio=0.8):
    random.shuffle(pairs)
    n_train = int(len(pairs) * train_ratio)
    train_pairs = pairs[:n_train]
    val_pairs = pairs[n_train:]
    for split, split_pairs in [('train', train_pairs), ('val', val_pairs)]:
        img_out = os.path.join(output_dir, 'images', split)
        lbl_out = os.path.join(output_dir, 'labels', split)
        os.makedirs(img_out, exist_ok=True)
        os.makedirs(lbl_out, exist_ok=True)
        for img_path, label_path in split_pairs:
            shutil.copy(img_path, os.path.join(img_out, os.path.basename(img_path)))
            shutil.copy(label_path, os.path.join(lbl_out, os.path.basename(label_path)))
    return len(train_pairs), len(val_pairs)

# Generate data.yaml from class_map.json
def generate_data_yaml(class_map_path, output_dir):
    with open(class_map_path, 'r') as f:
        class_map = json.load(f)
    names = class_map['id_to_label']
    nc = len(names)
    yaml_path = os.path.join(output_dir, 'data.yaml')
    with open(yaml_path, 'w') as f:
        f.write(f"train: {os.path.abspath(os.path.join(output_dir, 'images', 'train'))}\n")
        f.write(f"val: {os.path.abspath(os.path.join(output_dir, 'images', 'val'))}\n")
        f.write(f"\nnc: {nc}\n")
        f.write(f"names: {names}\n")
    print(f"data.yaml written to {yaml_path}")

if __name__ == "__main__":
    pairs = collect_all_pairs()
    print(f"Found {len(pairs)} image/label pairs.")
    n_train, n_val = split_and_copy(pairs, OUTPUT_DIR, TRAIN_RATIO)
    print(f"Split: {n_train} train, {n_val} val")
    class_map_path = os.path.join(YOLO_LABELS_DIR, 'class_map.json')
    generate_data_yaml(class_map_path, OUTPUT_DIR)
    print("Done!") 