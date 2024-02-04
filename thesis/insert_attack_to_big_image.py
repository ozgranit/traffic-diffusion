import copy
import os.path
import pickle

import cv2
import matplotlib.pyplot as plt
import numpy as np

from attacks.RFLA.attack_params import AttackParams
from load_images import load_annotations, crop_image


def load_stop_sign_coords():
    image_path = r'/workspace/traffic-diffusion/datasets/larger_images/image_outputs/road_1/dawn.jpg'
    annotation_path = 'datasets/larger_images/image_annotations/road_1.xml'
    annotations = load_annotations(annotation_path)
    for xmin, ymin, xmax, ymax, label in annotations:
        if label == "stop":  # Filter annotations with name "stop"
            return xmin, ymin, xmax, ymax
    return None

def insert_cropped_img_into_big_image(save_result:bool = False):
    cropped_img_path = r'/workspace/traffic-diffusion/datasets/imags_with_shadow/input/dawn_normal.png'
    large_image_path = r'/workspace/traffic-diffusion/datasets/larger_images/image_outputs/road_1/dawn.jpg'
    out_dir = r'/workspace/traffic-diffusion/datasets/imags_with_shadow/output'
    xmin, ymin, xmax, ymax = load_stop_sign_coords()

    cropped_img = cv2.imread(cropped_img_path, cv2.IMREAD_UNCHANGED)
    cropped_img = cv2.resize(cropped_img, (404, 427))
    large_image = cv2.imread(large_image_path, cv2.IMREAD_UNCHANGED)
    large_image[ymin:ymax, xmin:xmax] = cropped_img
    if save_result:
        out_path = os.path.join(out_dir, os.path.basename(cropped_img_path))
        cv2.imwrite(out_path, large_image)

    return large_image

def isolate_attack():
    src_clean_img_path = r'/workspace/traffic-diffusion/datasets/larger_images/image_inputs/road_1.jpg'
    src_clean_img = cv2.imread(src_clean_img_path, cv2.IMREAD_UNCHANGED)
    out_dir = r'/workspace/traffic-diffusion/datasets/imags_with_shadow/input_large_src_with_shadow'
    xmin, ymin, xmax, ymax = load_stop_sign_coords()
    cropped_src_clean_img = crop_image(src_clean_img, xmin, ymin, xmax, ymax)
    cropped_src_clean_img_resized = cv2.resize(cropped_src_clean_img, (224, 224))

    src_clean_img_image_raw = copy.deepcopy(cropped_src_clean_img_resized)
    attack_params_path = r'/workspace/traffic-diffusion/experiments/RFLA_tmp/larger_images/physical_attack_RFLA_LISA_isAdv-0_shape-hexagon_maxIter-200_ensemble-0_interploate-0/road_1/normal_attack/dawn.pkl'
    with open(attack_params_path, 'rb') as reader:
        attack_params = pickle.load(reader)
    cv2.fillPoly(src_clean_img_image_raw, attack_params.points,
                 (attack_params.red, attack_params.green, attack_params.blue))
    image_new = cv2.addWeighted(src_clean_img_image_raw, attack_params.alpha, cropped_src_clean_img_resized, 1 - attack_params.alpha, attack_params.gamma)
    image_new_resized = cv2.resize(image_new, cropped_src_clean_img.shape[:2][::-1])
    src_clean_img[ymin:ymax, xmin:xmax] = image_new_resized
    out_path = os.path.join(out_dir, os.path.basename(src_clean_img_path))
    cv2.imwrite(out_path, src_clean_img)

    # load cropped resized mask and save it in src orig size:
    mask_path = r'/tmp/pycharm_project_662/experiments/RFLA_tmp/larger_images/physical_attack_RFLA_LISA_isAdv-0_shape-hexagon_maxIter-200_ensemble-0_interploate-0/road_1/normal_attack/dawn_mask.png'
    out_mask_dir = r'/workspace/traffic-diffusion/datasets/imags_with_shadow/input_large_src_mask_attack'
    out_mask_path = os.path.join(out_mask_dir, os.path.basename(src_clean_img_path))
    mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
    mask = cv2.resize(mask, cropped_src_clean_img.shape[:2][::-1])
    mask_large_image = np.ones_like(src_clean_img) * 255
    mask_large_image[ymin:ymax, xmin:xmax] = mask

    cv2.imwrite(out_mask_path, mask_large_image)





    # generated_clean_img_path = r'/workspace/traffic-diffusion/datasets/larger_images/image_outputs/road_1/dawn.jpg'
    # gen_clean_img = cv2.imread(generated_clean_img_path, cv2.IMREAD_UNCHANGED)
    #
    # xmin, ymin, xmax, ymax = load_stop_sign_coords()
    # cropped_gen_clean_img = crop_image(gen_clean_img, xmin, ymin, xmax, ymax)
    # cropped_gen_clean_img = cv2.resize(cropped_gen_clean_img, (224, 224))
    #
    # attacked_cropped_img_path = r'/workspace/traffic-diffusion/datasets/imags_with_shadow/input/dawn_normal.png'
    # attacked_cropped_img = cv2.imread(attacked_cropped_img_path, cv2.IMREAD_UNCHANGED)
    #
    # # attacked_img = insert_cropped_img_into_big_image(save_result=False)
    # perturbation = attacked_cropped_img - cropped_gen_clean_img
    # # src_clean_img = cv2.resize(src_clean_img, attacked_img.shape[:2])
    # # src_img_with_attack = src_clean_img + perturbation

    # plt.imshow(image_new)
    # plt.show()
    #
    # plt.imshow(src_clean_img)
    # plt.show()

if __name__ == "__main__":
    # takes an attacked image - resize it and insert it back in original image
    # insert_cropped_img_into_big_image(save_result=True)

    # take attack from one image and add it to another
    isolate_attack()

