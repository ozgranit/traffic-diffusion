import json
import os
import cv2
import pickle
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
from collections import Counter
import sys

import numpy as np

sys.path.append('ShadowAttack')
# from ShadowAttack.shadow_attack import attack
# from ShadowAttack.utils import brightness, judge_mask_type, load_mask

# LISA stop sign label is: 12
# GTSRB stop sign label is: 14

def crop_image(image, xmin, ymin, xmax, ymax):
    return image[ymin:ymax, xmin:xmax]


def load_annotations(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    objects = root.findall("object")

    annotations = []
    for obj in objects:
        bbox = obj.find("bndbox")
        xmin = int(bbox.find("xmin").text)
        ymin = int(bbox.find("ymin").text)
        xmax = int(bbox.find("xmax").text)
        ymax = int(bbox.find("ymax").text)
        label = obj.find("name").text
        annotations.append((xmin, ymin, xmax, ymax, label))

    return annotations


def process_image(image_folder: str, annotation_folder: str, attack_db: str, crop_size: int = 32, mask_folder: str = None):
    file_names = []
    orig_imgs = []
    cropped_imgs = []
    cropped_resized_imgs = []
    labels = []
    bbx = []

    #for xml_file in os.listdir(annotation_folder):
    for xml_file in sorted(os.listdir(annotation_folder), key=lambda name: int(name.split('_')[1].split('.')[0])):  #os.listdir(annotation_folder):
        if xml_file.endswith('.xml'):
            img_file_name_without_ext = xml_file[:-4]
            image_filename = img_file_name_without_ext + '.png'
            image_path = os.path.join(image_folder, image_filename)
            if not os.path.exists(image_path):
                image_path = image_path[:-4] + '.jpg'
            annotation_path = os.path.join(annotation_folder, xml_file)

            if os.path.exists(image_path) and os.path.exists(annotation_path):
                image = cv2.imread(image_path)
                annotations = load_annotations(annotation_path)

                for xmin, ymin, xmax, ymax, label in annotations:
                    if label == "stop":  # Filter annotations with name "stop"
                        cropped_img = crop_image(image, xmin, ymin, xmax, ymax)
                        if mask_folder is not None:
                            image_mask = np.load(mask_folder + '/' + img_file_name_without_ext + '.npy')
                            image_mask = np.where(image_mask, 255, 0).astype(np.uint8)
                            cropped_mask = crop_image(image_mask, xmin, ymin, xmax, ymax)
                        label_value = 12 if attack_db == 'LISA' else 14

                        file_names.append(img_file_name_without_ext)
                        orig_imgs.append(image)
                        cropped_imgs.append(cropped_img)

                        # Resize cropped image to 32x32
                        cropped_resized = cv2.resize(cropped_img, (crop_size, crop_size))

                        cropped_resized_imgs.append(cropped_resized)
                        labels.append(label_value)
                        bbx.append([xmin, ymin, xmax, ymax])

    if mask_folder is not None:
        return file_names, orig_imgs, cropped_imgs, cropped_resized_imgs, labels, bbx, cropped_mask
    return file_names, orig_imgs, cropped_imgs, cropped_resized_imgs, labels, bbx


def plot_images(original, cropped):
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
    plt.title('Cropped Image')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

def plot_triple_images(original, cropped, resized_cropped):
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
    plt.title('Cropped Image')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(cv2.cvtColor(resized_cropped, cv2.COLOR_BGR2RGB))
    plt.title('Resized Cropped Image')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

def plot_triple_images_and_adv(original, cropped, resized_cropped, adv_img):
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 4, 1)
    plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 4, 2)
    plt.imshow(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
    plt.title('Cropped Image')
    plt.axis('off')

    plt.subplot(1, 4, 3)
    plt.imshow(cv2.cvtColor(resized_cropped, cv2.COLOR_BGR2RGB))
    plt.title('Resized Cropped Image')
    plt.axis('off')

    plt.subplot(1, 4, 4)
    plt.imshow(cv2.cvtColor(adv_img, cv2.COLOR_BGR2RGB))
    plt.title('Adv Image')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    attack_db = "LISA"  # # Replace with "LISA" or "GTSRB" depending on your use case, Replace with the actual attack database
    file_names, orig_imgs, cropped_imgs, cropped_resized_imgs, labels, bbx = process_image('larger_images/image_inputs',
                                                                'larger_images/image_annotations', attack_db, 32, 'larger_images/image_masks')
    index=2
    plot_triple_images(orig_imgs[index], cropped_imgs[index], cropped_resized_imgs[index])
    print("label {labels[index]}")
    print(f"orig img shape {orig_imgs[index].shape}, cropped img shape {cropped_imgs[index].shape}, cropped resized img shape {cropped_resized_imgs[index].shape}")
