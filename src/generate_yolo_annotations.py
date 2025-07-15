import os
import json
from glob import glob
from PIL import Image
import numpy as np
from tqdm import tqdm
import shutil

from utils.dsbi_utils import read_DSBI_annotation
from utils.angelina_utils import transform_angelina_label

# Helper: build a class mapping from all encountered labels
class ClassMap:
    def __init__(self):
        self.label_to_id = {}
        self.id_to_label = []
    def get_id(self, label):
        if label not in self.label_to_id:
            self.label_to_id[label] = len(self.id_to_label)
            self.id_to_label.append(label)
        return self.label_to_id[label]
    def save(self, path):
        with open(path, 'w') as f:
            json.dump({'id_to_label': self.id_to_label, 'label_to_id': self.label_to_id}, f, indent=2)

# --- DSBI ---
def process_dsbi(dsbi_root, output_dir, class_map):
    for root, dirs, files in os.walk(dsbi_root):
        for file in files:
            if file.endswith('+recto.jpg') or file.endswith('+verso.jpg'):
                img_path = os.path.join(root, file)
                label_path = img_path.replace('.jpg', '.txt')
                if not os.path.exists(label_path):
                    continue
                # Get image size
                with Image.open(img_path) as img:
                    width, height = img.size
                rects = read_DSBI_annotation(label_path, width, height, 0.0, False)
                yolo_lines = []
                for left, top, right, bottom, label in rects:
                    # Convert to YOLO format
                    x_center = (left + right) / 2 / width
                    y_center = (top + bottom) / 2 / height
                    w = (right - left) / width
                    h = (bottom - top) / height
                    class_id = class_map.get_id(label)
                    yolo_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}")
                # Write YOLO label file
                rel_dir = os.path.relpath(root, dsbi_root)
                out_dir = os.path.join(output_dir, 'DSBI', rel_dir)
                os.makedirs(out_dir, exist_ok=True)
                out_path = os.path.join(out_dir, os.path.splitext(file)[0] + '.txt')
                with open(out_path, 'w') as f:
                    f.write('\n'.join(yolo_lines))
                # Copy image file
                img_dst = os.path.join(out_dir, os.path.basename(img_path))
                if not os.path.exists(img_dst):
                    shutil.copy(img_path, img_dst)

# --- Angelina ---
def process_angelina(angelina_books_root, output_dir, class_map):
    for book in os.listdir(angelina_books_root):
        book_dir = os.path.join(angelina_books_root, book)
        if not os.path.isdir(book_dir):
            continue
        for file in os.listdir(book_dir):
            if file.endswith('.labeled.jpg'):
                img_path = os.path.join(book_dir, file)
                json_path = img_path.replace('.jpg', '.json')
                if not os.path.exists(json_path):
                    continue
                with Image.open(img_path) as img:
                    width, height = img.size
                with open(json_path, 'r') as f:
                    data = json.load(f)
                yolo_lines = []
                for shape in data['shapes']:
                    points = np.array(shape['points'])
                    x1, y1 = points[:,0].min(), points[:,1].min()
                    x2, y2 = points[:,0].max(), points[:,1].max()
                    x_center = ((x1 + x2) / 2) / width
                    y_center = ((y1 + y2) / 2) / height
                    w = (x2 - x1) / width
                    h = (y2 - y1) / height
                    label = transform_angelina_label(shape['label'])
                    class_id = class_map.get_id(label)
                    yolo_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}")
                # Write YOLO label file
                rel_dir = os.path.relpath(book_dir, angelina_books_root)
                out_dir = os.path.join(output_dir, 'Angelina', rel_dir)
                os.makedirs(out_dir, exist_ok=True)
                out_path = os.path.join(out_dir, os.path.splitext(file)[0] + '.txt')
                with open(out_path, 'w') as f:
                    f.write('\n'.join(yolo_lines))
                # Copy image file
                img_dst = os.path.join(out_dir, os.path.basename(img_path))
                if not os.path.exists(img_dst):
                    shutil.copy(img_path, img_dst)

if __name__ == "__main__":
    # Set dataset roots
    dsbi_root = os.path.join('dataset', 'DSBI', 'DSBI', 'data')
    angelina_books_root = os.path.join('dataset', 'AngelinaDataset', 'AngelinaDataset', 'books')
    output_dir = os.path.join('yolo_labels')
    os.makedirs(output_dir, exist_ok=True)
    class_map = ClassMap()
    print('Processing DSBI...')
    process_dsbi(dsbi_root, output_dir, class_map)
    print('Processing Angelina...')
    process_angelina(angelina_books_root, output_dir, class_map)
    # Save class mapping
    class_map.save(os.path.join(output_dir, 'class_map.json'))
    print('Done! YOLO annotation files and class mapping saved to', output_dir) 