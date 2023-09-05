import os
import pickle

import cv2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
# import cv2

dataset = 'GTSRB'
# pkl_file_path = 'dataset/LISA/test.pkl'
pkl_file_path = f'dataset/{dataset}/test.pkl'

with open(pkl_file_path, 'rb') as pkl_file:
    data_dict = pickle.load(pkl_file)
#     print(image_data)
LISA_STOP_SIGN_LABEL = 12
GTSRB_STOP_SIGN_LABEL = 14
image_data = data_dict['data']  # Assuming 'data' key contains the image data

labels = data_dict['labels']  # Assuming 'labels' key contains the labels
print(len(image_data))
# Determine the number of rows and columns for the subplots
num_images_to_plot = 5
num_of_imgs_to_display = 10
num_rows = num_of_imgs_to_display // num_images_to_plot
if len(image_data) % num_images_to_plot != 0:
    num_rows += 1
cnt = 0
orig_label = LISA_STOP_SIGN_LABEL if dataset == 'LISA' else GTSRB_STOP_SIGN_LABEL
# Create subplots
fig, axes = plt.subplots(num_rows, num_images_to_plot, figsize=(15, 5 * num_rows))
print("starting...")
for image_index, (image_array, label) in enumerate(zip(image_data, labels)):
    if label == orig_label:
        image = Image.fromarray(np.uint8(image_array))

        row = cnt // num_images_to_plot
        col = cnt % num_images_to_plot
        #     print(image.size)
        axes[row, col].imshow(cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB))
        axes[row, col].set_title(f'Label: {label}')
        axes[row, col].axis('off')
        cnt += 1
        if cnt == 10:
            break
    #     image.show()
    #     print("Label:", label)

# Remove any empty subplots
for i in range(num_of_imgs_to_display, num_rows * num_images_to_plot):
    row = i // num_images_to_plot
    col = i % num_images_to_plot
    axes[row, col].axis('off')

plt.tight_layout()
plt.show()
print("done!")


