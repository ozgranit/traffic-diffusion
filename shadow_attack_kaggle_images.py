import json
import os
import cv2
import pickle
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
from collections import Counter
import sys

from load_kaggle_images import process_image, plot_triple_images_and_adv

sys.path.append('ShadowAttack')
from ShadowAttack.shadow_attack import attack
from ShadowAttack.utils import brightness, judge_mask_type, load_mask

# LISA stop sign label is: 12
# GTSRB stop sign label is: 14

def load_params():
    with open('ShadowAttack/params.json', 'rb') as f:
        params = json.load(f)
        class_n_gtsrb = params['GTSRB']['class_n']
        class_n_lisa = params['LISA']['class_n']
        device = params['device']
        position_list, mask_list = load_mask()

        return position_list, mask_list

def attack_digital(attack_db: str = "LISA"):
    shadow_level = 0.5  # Replace with the desired shadow level
    position_list, mask_list = load_params()
    save_dir = f'./adv_img/{attack_db}/{int(shadow_level * 100)}'
    try:
        os.makedirs(save_dir, exist_ok=True)
    except FileExistsError:
        for name in os.listdir(save_dir):
            os.remove(os.path.join(save_dir, name))

    save_dir_bmp = os.path.join(save_dir, 'bmp')
    save_dir_png = os.path.join(save_dir, 'png')
    os.makedirs(save_dir_bmp)
    os.makedirs(save_dir_png)

    file_names, orig_imgs, cropped_imgs, cropped_resized_imgs, labels = process_image('kaggle_images', #/workspace/traffic-diffusion/
                                                            'kaggle_annotations', attack_db) #/workspace/traffic-diffusion/
    cnt_attacked = 0
    print(f"Start attacking {len(cropped_resized_imgs)} images")
    for index in range(len(cropped_resized_imgs)):
        mask_type = judge_mask_type(attack_db, labels[index])
        if brightness(cropped_resized_imgs[index], mask_list[mask_type]) >= 120:
            adv_img, success, num_query = attack(
                cropped_resized_imgs[index], labels[index], position_list[mask_type])

            cv2.imwrite(f"{save_dir_bmp}/{file_names[index]}_{labels[index]}_{num_query}_{success}.bmp", adv_img)
            cv2.imwrite(f"{save_dir_png}/{file_names[index]}_{labels[index]}_{num_query}_{success}.png", adv_img)
            cnt_attacked+=1

            # Plot one or two examples
            if cnt_attacked < 2:
                plot_triple_images_and_adv(orig_imgs[index], cropped_imgs[index], cropped_resized_imgs[index], adv_img)
        else:
            print(f"Skip index {index} because of brightness")

    print("Attack finished! Success rate: ", end='')
    print(Counter(map(lambda x: x[:-4].split('_')[-1],
                      os.listdir(save_dir)))['True'] / len(os.listdir(save_dir)))



if __name__ == "__main__":
    # Call the attack_digital() function
    attack_db = "GTSRB"  # # Replace with "LISA" or "GTSRB" depending on your use case, Replace with the actual attack database
    attack_digital(attack_db)