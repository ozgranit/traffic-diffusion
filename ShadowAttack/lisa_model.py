# -*- coding: utf-8 -*-

import gc
import os
from typing import Union

import cv2
import time
import pickle
import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms
from torch.utils.data.dataloader import DataLoader

# from config import PARAMS_PATH
# from utils import SmoothCrossEntropyLoss
# from utils import draw_shadow
# from utils import shadow_edge_blur
# from utils import judge_mask_type
# from utils import load_mask

from ShadowAttack.config import PARAMS_PATH, MODEL_PATH
from ShadowAttack.model_base import ModelBase
from ShadowAttack.utils import SmoothCrossEntropyLoss
from ShadowAttack.utils import draw_shadow
from ShadowAttack.utils import shadow_edge_blur
from ShadowAttack.utils import judge_mask_type
from ShadowAttack.utils import load_mask
# with open('ShadowAttack/params.json', 'r') as config:

class LisaCNN(nn.Module):

    def __init__(self, n_class):

        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, (8, 8), stride=(2, 2), padding=3)
        self.conv2 = nn.Conv2d(64, 128, (6, 6), stride=(2, 2), padding=0)
        self.conv3 = nn.Conv2d(128, 128, (5, 5), stride=(1, 1), padding=0)
        self.fc = nn.Linear(512, n_class)

    def forward(self, x):

        x = nn.ReLU()(self.conv1(x))
        x = nn.ReLU()(self.conv2(x))
        x = nn.ReLU()(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class LisaModel(ModelBase):
    def __init__(self, adv_model: bool=False, crop_size: tuple[int, int] = (32, 32)):
        self.model_name = 'LISA'
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'  # params['device']
        self.crop_size = crop_size
        self.load_params(self.model_name)
        self.model = self.load_model(adv_model)

    def load_model(self, adv_model: bool = False):
        model = LisaCNN(n_class=self.class_n).to(self.device)
        model.load_state_dict(
            torch.load(f'{MODEL_PATH}/{"adv_" if adv_model else ""}model_{self.model_name.lower()}.pth',
                       map_location=torch.device(self.device)))
        model.eval()

        return model

    def test_single_image(self, img: Union[str, np.ndarray], ground_truth: int = None):
        msg = ''
        img = self.load_img_if_needed(img)
        # img is already loaded
        img = self.pre_process_image(img, crop_size=self.crop_size, device=self.device)
        confidence, pred_label = self.predict_img(img)

        if ground_truth:
            print(f'Correct: {pred_label==ground_truth}', end=' ')
            print(f'Predict: {pred_label} Confidence: {confidence*100}%')
            msg = f'Correct: {pred_label==ground_truth}\nPredict: {pred_label} Confidence: {confidence*100}%'

            return pred_label, pred_label == ground_truth, msg

        return pred_label, None, ''

    @staticmethod
    def pre_process_image(img: np.ndarray, crop_size: tuple[int, int] = (32, 32), device: str = 'cpu') -> torch.tensor:
        img = cv2.resize(img, crop_size)
        img = transforms.ToTensor()(img)
        img = img.unsqueeze(0).to(device)

        return img


if __name__ == '__main__':

    # model training
    # train_model(adv_train=False)

    # model testing
    # test_model(adv_model=False)

    # test a single image
    lisa_model = LisaModel(False, (32, 32))
    lisa_model.test_single_image('./tmp/lisa_30.jpg', 9, adv_model=False)
