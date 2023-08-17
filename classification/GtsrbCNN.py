import json
import torch
from torch import nn

from classification.utils import load_img
from utils import MODELS_PATH


class GtsrbCNN(nn.Module):
    def __init__(self, n_class):

        super().__init__()
        self.model_name = 'GTSRB'
        self.n_class = n_class
        self.device = None
        self.init_params()
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

    def init_params(self):
        with open(MODELS_PATH + 'params.json', 'r') as config:
            params = json.load(config)
            self.n_class = params[self.model_name]['class_n']
            self.device = params['device']
            # position_list, _ = load_mask()

    def test_single_image(self, img_path, label, adv_model=False):
        trained_model = GtsrbCNN(n_class=self.n_class).to(self.device)
        trained_model.load_state_dict(
            torch.load(MODELS_PATH + f'{"adv_" if adv_model else ""}model_gtsrb.pth',
                       map_location=torch.device(self.device)))
        trained_model.eval()

        img = load_img(self.device, img_path)

        predict = torch.softmax(trained_model(img)[0], 0)
        index = int(torch.argmax(predict).data)
        confidence = float(predict[index].data)

        print(f'Correct: {index==label}', end=' ')
        print(f'Predict: {index} Confidence: {confidence*100}%')

        return index, index == label

