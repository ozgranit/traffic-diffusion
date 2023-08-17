import json

import cv2
import torch
from torch import nn
from torchvision import transforms

from utils import MODELS_PATH, load_img


class LisaCNN(nn.Module):

    def __init__(self, n_class):

        super().__init__()
        self.model_name = 'LISA'
        self.n_class = n_class
        self.device = None
        self.init_params()
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

    def init_params(self):
        with open(MODELS_PATH + 'params.json', 'r') as config:
            params = json.load(config)
            self.class_n = params['LISA']['class_n']
            self.device = params['device']
            # position_list, _ = load_mask()

    def test_single_image(self, img_path, ground_truth, adv_model=False):
        trained_model = LisaCNN(n_class=self.n_class).to(self.device)
        trained_model.load_state_dict(
            torch.load(MODELS_PATH + f'{"adv_" if adv_model else ""}model_lisa.pth',
                       map_location=torch.device(self.device)))
        trained_model.eval()

        img = load_img(self.device, img_path)


        predict = torch.softmax(trained_model(img)[0], 0)
        index = int(torch.argmax(predict).data)
        confidence = float(predict[index].data)

        print(f'Correct: {index==ground_truth}', end=' ')
        print(f'Predict: {index} Confidence: {confidence*100}%')

        return index, index == ground_truth