import pickle

import cv2
import numpy as np
from matplotlib import pyplot as plt

from load_images import plot_images, process_image


def check_octagon_mask():

    attack_db = "LISA"  # # Replace with "LISA" or "GTSRB" depending on your use case, Replace with the actual attack database
    file_names, orig_imgs, cropped_imgs, cropped_resized_imgs, labels, bbx = process_image('kaggle_images',
                                                                                      'kaggle_annotations', attack_db)

    ind = file_names.index("road66")

    mask_path = r'ShadowAttack/mask/octagon.pkl'
    with open(mask_path, 'rb') as pkl_file:
        mask_im = pickle.load(pkl_file)
    # plt.imshow(mask_im)

    # mask_im = 255-mask_im
    img = cropped_imgs[ind]
    cv2.imwrite("octagon_mask.png", mask_im)
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title('cropped_img')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(mask_im, cmap='gray')
    plt.title('mask')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

def check_larger_images_mask():
    mask_path = r'larger_images/image_masks/road_1.npy'
    image_mask = np.load(mask_path)
    plt.imshow(image_mask, cmap='gray')
    plt.show()

if __name__ == "__main__":
    check_larger_images_mask()
