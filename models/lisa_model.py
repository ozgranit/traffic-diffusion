from typing import Tuple
import cv2
import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms
from attacks.ShadowAttack.shadow_attack_settings import MODEL_PATH
from models.base_model import BaseModel
from settings import LISA, DEVICE
import torch.nn.functional as F


class LisaCNN(nn.Module):
    def __init__(self, n_class, interpolate_size: int = None):

        super().__init__()
        self.interpolate_size = interpolate_size
        self.conv1 = nn.Conv2d(3, 64, (8, 8), stride=(2, 2), padding=3)
        self.conv2 = nn.Conv2d(64, 128, (6, 6), stride=(2, 2), padding=0)
        self.conv3 = nn.Conv2d(128, 128, (5, 5), stride=(1, 1), padding=0)
        self.fc = nn.Linear(512, n_class)

    def forward(self, x):
        if self.interpolate_size is not None:
            # Resize the input tensor using interpolate to the specified size
            x = F.interpolate(x, size=self.interpolate_size, mode='bilinear', align_corners=False)

        x = nn.ReLU()(self.conv1(x))
        x = nn.ReLU()(self.conv2(x))
        x = nn.ReLU()(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class LisaModel(BaseModel):
    def __init__(self, adv_model: bool=False, crop_size: int = 32, use_interpolate: bool = False):
        super().__init__(crop_size, use_interpolate)
        self.model_name = LISA
        self.load_params(self.model_name)
        self.model = self.load_model(adv_model)

    def load_model(self, adv_model: bool = False):
        model = LisaCNN(n_class=self.class_n, interpolate_size=self.crop_size).to(self.device)
        model.load_state_dict(
            torch.load(f'{MODEL_PATH}/{"adv_" if adv_model else ""}model_{self.model_name.lower()}.pth',
                       map_location=torch.device(self.device)))
        model.eval()

        return model

    @staticmethod
    def pre_process_image(img: np.ndarray, crop_size: Tuple[int, int] = (32, 32), device: str = DEVICE, use_interpolate: bool = False) -> torch.tensor:
        if not use_interpolate:
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
    lisa_model = LisaModel(False, 32, use_interpolate=False)
    lisa_model.test_single_image('./tmp/lisa_30.jpg', 9)
