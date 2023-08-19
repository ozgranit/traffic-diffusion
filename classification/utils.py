from torchvision import transforms
import numpy as np
import cv2
import torch
import json

from classification.GtsrbCNN import GtsrbCNN
from classification.LisaCNN import LisaCNN
MODELS_PATH = 'classification/classification_models/'
LISA_GROUND_TRUTH = 12
GTSRB_GROUND_TRUTH = 14

class Utils:
    def __init__(self):
        self.models = {}
        self.device = None
        self.init_models()

    def init_models(self):
        with open(MODELS_PATH + 'params.json', 'r') as config:
            params = json.load(config)
            class_n_gtsrb = params['GTSRB']['class_n']
            class_n_lisa = params['LISA']['class_n']
            self.device = params['device']

        gtsrbCNN = GtsrbCNN(class_n_gtsrb, GTSRB_GROUND_TRUTH, False).to(self.device)
        gtsrbCNN.load_state_dict(
            torch.load(MODELS_PATH + 'model_gtsrb.pth',
                       map_location=torch.device(self.device)))
        gtsrbCNN.eval()

        gtsrbCNN_adv = GtsrbCNN(class_n_gtsrb, GTSRB_GROUND_TRUTH, True).to(self.device)
        gtsrbCNN_adv.load_state_dict(
            torch.load(MODELS_PATH + 'adv_model_gtsrb.pth',
                       map_location=torch.device(self.device)))
        gtsrbCNN_adv.eval()

        lisaCNN = LisaCNN(class_n_lisa, LISA_GROUND_TRUTH, False).to(self.device)
        lisaCNN.load_state_dict(
            torch.load(MODELS_PATH + 'model_lisa.pth',
                       map_location=torch.device(self.device)))
        lisaCNN.eval()

        lisaCNN_adv = LisaCNN(class_n_lisa, LISA_GROUND_TRUTH, True).to(self.device)
        lisaCNN_adv.load_state_dict(
            torch.load(MODELS_PATH + 'adv_model_lisa.pth',
                       map_location=torch.device(self.device)))
        lisaCNN_adv.eval()

        self.models['gtsrb'] = gtsrbCNN
        self.models['gtsrb_adv'] = gtsrbCNN_adv
        self.models['lisa'] = lisaCNN
        self.models['lisa_adv'] = lisaCNN_adv

    def test_single_image(self, img_path, img_name, model_name, adv_model=False):
        model = self.models[model_name.lower() + f'{"_adv" if adv_model else ""}']
        path = os.path.join(img_path, img_name)

        img = cv2.imread(path)
        img = cv2.resize(img, (32, 32))
        if model_name.lower() == 'gtsrb':
            img = pre_process_image(img).astype(np.float32)
        img = transforms.ToTensor()(img)
        img = img.unsqueeze(0).to(self.device)

        predict = torch.softmax(model(img)[0], 0)
        index = int(torch.argmax(predict).data)
        confidence = float(predict[index].data)

        # print(f'Correct: {index==ground_truth}', end=' ')
        # print(f'Predict: {index} Confidence: {confidence*100}%')

        result = {
            'name': img_name,
            'path': img_path,
            'model': model_name + f'{"_adv" if adv_model else ""}',
            'true_label': model.ground_truth,
            'pred_label': index,
            'success': index == model.ground_truth,
            'pred_score': confidence
        }

        return result


def pre_process_image(image):
    image[:, :, 0] = cv2.equalizeHist(image[:, :, 0])
    image[:, :, 1] = cv2.equalizeHist(image[:, :, 1])
    image[:, :, 2] = cv2.equalizeHist(image[:, :, 2])
    image = image / 255. - .5
    return image


def transform_image(image, ang_range, shear_range, trans_range, preprocess):
    # Rotation
    ang_rot = np.random.uniform(ang_range) - ang_range / 2
    rows, cols, ch = image.shape
    rot_m = cv2.getRotationMatrix2D((cols / 2, rows / 2), ang_rot, 1)

    # Translation
    tr_x = trans_range * np.random.uniform() - trans_range / 2
    tr_y = trans_range * np.random.uniform() - trans_range / 2
    trans_m = np.float32([[1, 0, tr_x], [0, 1, tr_y]])

    # Shear
    pts1 = np.float32([[5, 5], [20, 5], [5, 20]])

    pt1 = 5 + shear_range * np.random.uniform() - shear_range / 2
    pt2 = 20 + shear_range * np.random.uniform() - shear_range / 2

    pts2 = np.float32([[pt1, 5], [pt2, pt1], [5, pt2]])

    shear_m = cv2.getAffineTransform(pts1, pts2)

    image = cv2.warpAffine(image, rot_m, (cols, rows))
    image = cv2.warpAffine(image, trans_m, (cols, rows))
    image = cv2.warpAffine(image, shear_m, (cols, rows))

    image = pre_process_image(image) if preprocess else image

    return image


def load_img(device, img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (32, 32))
    img = pre_process_image(img).astype(np.float32)
    img = transforms.ToTensor()(img)
    img = img.unsqueeze(0).to(device)
    return img