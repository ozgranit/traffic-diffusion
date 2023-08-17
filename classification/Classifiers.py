import gc
import pickle
import time
import json
import cv2
import os
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data.dataloader import DataLoader

from classification.GtsrbCNN import test_single_image_gtsrb
from classification.LisaCNN import test_single_image_lisa
from classification.utils import MODELS_PATH

with open(MODELS_PATH + 'params.json', 'r') as config:
    params = json.load(config)
    class_n_gtsrb = params['GTSRB']['class_n']
    device = params['device']
    # position_list, _ = load_mask()

with open(MODELS_PATH + 'params.json', 'r') as config:
    params = json.load(config)
    class_n_lisa = params['LISA']['class_n']
    device = params['device']
    # position_list, _ = load_mask()

if __name__ == '__main__':
    directory = './'
    for filename in os.listdir(directory):
        if filename.lower().endswith(('.png', '.jpg')):
            image_path = os.path.join(directory, filename)
            print(image_path)
            test_single_image_gtsrb(image_path, 14, adv_model=False)
            test_single_image_gtsrb(image_path, 14, adv_model=True)
            test_single_image_lisa(image_path, 12, adv_model=False)
            test_single_image_lisa(image_path, 12, adv_model=True)
            print()

