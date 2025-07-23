import os
import glob
import random
import shutil
import xml.etree.ElementTree as ET

# Paths
IMAGES_DIR = 'frames'
ANNOTATIONS_DIR = 'annotations'
OUTPUT_DIR = 'dataset'
TRAIN_RATIO = 0.8

# Classes (edit as needed)
CLASSES = ['Player 1', 'Player 2', 'Shuttle']

def voc_to_yolo(xml_file, img_w, img_h):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    yolo_lines = []
    for obj in root.findall('object'):
        cls = obj.find('name').text
        print(f"Found class: '{cls}'")  # Add this line
        if cls not in CLASSES:
            print(f"Skipping class: '{cls}' (not in CLASSES)")
            continue
        cls_id = CLASSES.index(cls)
        bndbox = obj.find('bndbox')
        xmin = float(bndbox.find('xmin').text)
        ymin = float(bndbox.find('ymin').text)
        xmax = float(bndbox.find('xmax').text)
        ymax = float(bndbox.find('ymax').text)
        # Convert to YOLO format
        x_center = ((xmin + xmax) / 2) / img_w
        y_center = ((ymin + ymax) / 2) / img_h
        width = (xmax - xmin) / img_w
        height = (ymax - ymin) / img_h
        yolo_lines.append(f"{cls_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
    return yolo_lines

def get_image_size(img_path):
    import cv2
    img = cv2.imread(img_path)
    h, w = img.shape[:2]
    return w, h

def main():
    # Gather all annotation files
    xml_files = sorted(glob.glob(os.path.join(ANNOTATIONS_DIR, '*.xml')))
    img_files = [os.path.join(IMAGES_DIR, os.path.splitext(os.path.basename(x))[0] + '.jpg') for x in xml_files]
    # Filter only those where both image and xml exist
    pairs = [(img, xml) for img, xml in zip(img_files, xml_files) if os.path.exists(img)]
    print(f"Found {len(pairs)} image-annotation pairs.")
    # Shuffle and split
    random.shuffle(pairs)
    split_idx = int(len(pairs) * TRAIN_RATIO)
    train_pairs = pairs[:split_idx]
    val_pairs = pairs[split_idx:]
    # Prepare output folders
    for split, split_pairs in zip(['train', 'val'], [train_pairs, val_pairs]):
        img_out_dir = os.path.join(OUTPUT_DIR, 'images', split)
        label_out_dir = os.path.join(OUTPUT_DIR, 'labels', split)
        os.makedirs(img_out_dir, exist_ok=True)
        os.makedirs(label_out_dir, exist_ok=True)
        for img_path, xml_path in split_pairs:
            # Copy image
            shutil.copy(img_path, img_out_dir)
            # Convert annotation
            w, h = get_image_size(img_path)
            yolo_lines = voc_to_yolo(xml_path, w, h)
            label_name = os.path.splitext(os.path.basename(img_path))[0] + '.txt'
            with open(os.path.join(label_out_dir, label_name), 'w') as f:
                f.write('\n'.join(yolo_lines) + '\n')
    print('Conversion and split complete!')
    print(f"Train images: {len(train_pairs)}, Val images: {len(val_pairs)}")
    print(f"YOLO dataset is ready in '{OUTPUT_DIR}' folder.")

if __name__ == '__main__':
    main() 