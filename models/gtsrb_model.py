from typing import Union, Tuple
import cv2
import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms
from attacks.ShadowAttack.shadow_attack_settings import MODEL_PATH
from models.base_model import BaseModel
from settings import GTSRB


class GtsrbCNN(nn.Module):

    def __init__(self, n_class):

        super().__init__()
        self.color_map = nn.Conv2d(3, 3, (1, 1), stride=(1, 1), padding=0)
        self.module1 = nn.Sequential(
            nn.Conv2d(3, 32, (5, 5), stride=(1, 1), padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 32, (5, 5), stride=(1, 1), padding=2),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.ReLU(),
            nn.Dropout(p=0.5),
        )
        self.module2 = nn.Sequential(
            nn.Conv2d(32, 64, (5, 5), stride=(1, 1), padding=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, (5, 5), stride=(1, 1), padding=2),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.ReLU(),
            nn.Dropout(p=0.5),
        )
        self.module3 = nn.Sequential(
            nn.Conv2d(64, 128, (5, 5), stride=(1, 1), padding=2),
            nn.ReLU(),
            nn.Conv2d(128, 128, (5, 5), stride=(1, 1), padding=2),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.ReLU(),
            nn.Dropout(p=0.5),
        )
        self.fc1 = nn.Sequential(
            nn.Linear(14336, 1024, bias=True),
            nn.ReLU(),
            nn.Dropout(p=0.5)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(1024, 1024, bias=True),
            nn.ReLU(),
            nn.Dropout(p=0.5),
        )
        self.fc3 = nn.Linear(1024, n_class, bias=True)

    def forward(self, x):

        x = self.color_map(x)
        branch1 = self.module1(x)
        branch2 = self.module2(branch1)
        branch3 = self.module3(branch2)

        branch1 = branch1.reshape(branch1.shape[0], -1)
        branch2 = branch2.reshape(branch2.shape[0], -1)
        branch3 = branch3.reshape(branch3.shape[0], -1)
        concat = torch.cat([branch1, branch2, branch3], 1)

        out = self.fc1(concat)
        out = self.fc2(out)
        out = self.fc3(out)
        return out

class GtsrbModel(BaseModel):
    def __init__(self, adv_model: bool=False, crop_size: Tuple[int, int] = (32, 32)):
        self.model_name = GTSRB
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'  # params['device']
        self.crop_size = crop_size
        self.load_params(self.model_name)
        self.model = self.load_model(adv_model)

    def load_model(self, adv_model: bool = False):
        model = GtsrbCNN(n_class=self.class_n).to(self.device)
        model.load_state_dict(
            torch.load(f'{MODEL_PATH}/{"adv_" if adv_model else ""}model_{self.model_name.lower()}.pth',
                       map_location=torch.device(self.device)))
        model.eval()

        return model

    @staticmethod
    def pre_process_image(img: np.ndarray, crop_size: tuple[int, int] = (32, 32), device: str = 'cpu') -> torch.tensor:
        img = cv2.resize(img, crop_size)

        img[:, :, 0] = cv2.equalizeHist(img[:, :, 0])
        img[:, :, 1] = cv2.equalizeHist(img[:, :, 1])
        img[:, :, 2] = cv2.equalizeHist(img[:, :, 2])
        img = img / 255. - .5
        img = img.astype(np.float32)

        img = transforms.ToTensor()(img)
        img = img.unsqueeze(0).to(device)

        return img

if __name__ == '__main__':

    # model training
    # train_model(adv_train=False)

    # model testing
    # test_model(adv_model=False)

    # test a single image
    gtsrb_model = GtsrbModel(False, (32, 32))
    gtsrb_model.test_single_image('./tmp/lisa_30.jpg', 9, adv_model=False)
