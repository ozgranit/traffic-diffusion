import json
import cv2
import torch
from torch import nn
from torchvision import transforms

class LisaCNN(nn.Module):

    def __init__(self, n_class, ground_truth, adv_model):
        super().__init__()
        self.model_name = 'LISA'
        self.n_class = n_class
        self.adv_model = adv_model
        self.ground_truth = ground_truth
        self.name = 'LISA' + f'{"_adv" if adv_model else ""}'

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