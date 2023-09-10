import json
import os
from typing import Union, Tuple

import cv2
import numpy as np
from abc import ABC, abstractmethod
import torch

from attacks.ShadowAttack.shadow_attack_settings import PARAMS_PATH
from attacks.ShadowAttack.utils import load_mask


class BaseModel(ABC):

    def load_params(self, model_name: str):
        """

        Args:
            model_name: can be 'LISA' or 'GTSRB'

            loads self.params, self.class_n
        """
        with open(PARAMS_PATH, 'r') as config:
            self.params = json.load(config)
            self.class_n = self.params[model_name.upper()]['class_n']
            self.position_list, _ = load_mask()

    def test_single_image(self, img: Union[str, np.ndarray],
                          ground_truth: int = None,
                          print_results: bool = True,
                          description: str = ''):
        with torch.no_grad():
            msg = ''
            img = self.load_img_if_needed(img)
            # img is already loaded
            img = self.pre_process_image(img, crop_size=(self.crop_size, self.crop_size), device=self.device)
            confidence, pred_label = self.__predict_img(img)

            if ground_truth:
                if print_results:
                    print(description)
                    print(f'Correct: {pred_label==ground_truth}', end=' ')
                    print(f'Predict: {pred_label} Confidence: {confidence*100}%')
                msg = f'Correct: {pred_label==ground_truth}\nPredict: {pred_label} Confidence: {confidence*100}%'

                return confidence, pred_label, pred_label == ground_truth, msg

            return confidence, pred_label, None, msg

    def __predict_img(self, img):
        predict = torch.softmax(self.model(img)[0], 0)
        pred_label = int(torch.argmax(predict).data)
        confidence = float(predict[pred_label].data)

        return confidence, pred_label

    def load_img_if_needed(self, img: Union[str, np.ndarray]):
        # If img is path
        if isinstance(img, str):
            if os.path.exists(img):
                img = cv2.imread(img)
            else:
                raise FileExistsError(img)
        return img

    @abstractmethod
    def pre_process_image(self, img: np.ndarray, crop_size: Tuple[int, int] = (32, 32), device: str = 'cpu') -> torch.tensor:
        # Implement pre_process_image
        pass