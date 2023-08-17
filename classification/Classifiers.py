import json
import os
from GtsrbCNN import GtsrbCNN
from LisaCNN import LisaCNN
# from utils import MODELS_PATH
# import numpy as np
# import torch
# import torch.nn as nn
# from torchvision import transforms
# from torch.utils.data.dataloader import DataLoader

from utils import MODELS_PATH

IMAGES_PATH = 'kaggle_images/'
LISA_GROUND_TRUTH = 12
GTSRB_GROUND_TRUTH = 14

if __name__ == '__main__':

    with open(MODELS_PATH + 'params.json', 'r') as config:
        params = json.load(config)
        class_n_gtsrb = params['GTSRB']['class_n']
        class_n_lisa = params['LISA']['class_n']
        device = params['device']

    gtsrbCNN = GtsrbCNN(n_class=class_n_gtsrb).to(device)
    lisaCNN = LisaCNN(n_class=class_n_lisa).to(device)
    for filename in os.listdir(IMAGES_PATH):
        if filename.lower().endswith(('.png', '.jpg')):
            print("-------------------File name: ", filename)
            image_path = os.path.join(IMAGES_PATH, filename)
            # print("GTSRB adv_model=False")
            #gtsrbCNN.test_single_image(image_path, GTSRB_GROUND_TRUTH, adv_model=False)
            # print("GTSRB adv_model=True")
            #gtsrbCNN.test_single_image(image_path, GTSRB_GROUND_TRUTH, adv_model=True)
            print("LISA adv_model=False")
            lisaCNN.test_single_image(image_path, LISA_GROUND_TRUTH, adv_model=False)
            print("LISA adv_model=True")
            lisaCNN.test_single_image(image_path, LISA_GROUND_TRUTH, adv_model=True)
            print()