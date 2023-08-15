import os
import cv2
import pickle
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
from collections import Counter


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


def process_image(image_folder, annotation_folder, flag):
    orig_imgs = []
    cropped_imgs = []
    cropped_resized_imgs = []
    labels = []

    for xml_file in os.listdir(annotation_folder):
        if xml_file.endswith('.xml'):
            image_filename = xml_file[:-4] + '.png'
            image_path = os.path.join(image_folder, image_filename)
            annotation_path = os.path.join(annotation_folder, xml_file)

            if os.path.exists(image_path) and os.path.exists(annotation_path):
                image = cv2.imread(image_path)
                annotations = load_annotations(annotation_path)

                for xmin, ymin, xmax, ymax, label in annotations:
                    if label == "stop":  # Filter annotations with name "stop"
                        cropped_img = crop_image(image, xmin, ymin, xmax, ymax)
                        label_value = '12' if flag == 'LISA' else '14'

                        orig_imgs.append(image)
                        cropped_imgs.append(cropped_img)

                        # Resize cropped image to 32x32
                        cropped_resized = cv2.resize(cropped_img, (32, 32))

                        cropped_resized_imgs.append(cropped_resized)
                        labels.append(label_value)

    return orig_imgs, cropped_imgs, cropped_resized_imgs, labels


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


def attack_digital():
    attack_db = "your_attack_db"  # Replace with the actual attack database
    shadow_level = 0.5  # Replace with the desired shadow level
    flag = "LISA"  # Replace with "LISA" or "GTSRB" depending on your use case

    mask_list = {
        "mask_type_1": mask1,  # Replace with actual mask functions
        "mask_type_2": mask2,
        # Add more mask types and corresponding functions as needed
    }

    position_list = {
        "mask_type_1": position1,  # Replace with actual position functions
        "mask_type_2": position2,
        # Add more mask types and corresponding functions as needed
    }

    save_dir = f'./adv_img/{attack_db}/{int(shadow_level * 100)}'
    try:
        os.makedirs(save_dir, exist_ok=True)
    except FileExistsError:
        for name in os.listdir(save_dir):
            os.remove(os.path.join(save_dir, name))

    original_images, cropped_images, labels = process_image('/workspace/traffic-diffusion/kaggle_images',
                                                            '/workspace/traffic-diffusion/kaggle_annotations', flag)

    for index in range(len(original_images)):
        mask_type = judge_mask_type(attack_db, labels[index])
        if brightness(original_images[index], mask_list[mask_type]) >= 120:
            adv_img, success, num_query = attack(
                original_images[index], labels[index], position_list[mask_type])
            cv2.imwrite(f"{save_dir}/{index}_{labels[index]}_{num_query}_{success}.bmp", adv_img)

            # Plot one or two examples
            if index < 2:
                plot_images(original_images[index], cropped_images[index])

    print("Attack finished! Success rate: ", end='')
    print(Counter(map(lambda x: x[:-4].split('_')[-1],
                      os.listdir(save_dir)))['True'] / len(os.listdir(save_dir)))


# Call the attack_digital() function
# attack_digital()

flag='LISA'
orig_imgs, cropped_imgs, cropped_resized_imgs, labels = process_image('kaggle_images',
                                                            'kaggle_annotations', flag)
index=0
plot_triple_images(orig_imgs[index], cropped_imgs[index], cropped_resized_imgs[index])
print("label {labels[index]}")
print(f"orig img shape {orig_imgs[index].shape}, cropped img shape {cropped_imgs[index].shape}, cropped resized img shape {cropped_resized_imgs[index].shape}")