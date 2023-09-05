from typing import Union
import cv2
import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms
from attacks.ShadowAttack.shadow_attack_settings import MODEL_PATH
from models.base_model import BaseModel
from settings import LISA


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

class LisaModel(BaseModel):
    def __init__(self, adv_model: bool=False, crop_size: tuple[int, int] = (32, 32)):
        self.model_name = LISA
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
