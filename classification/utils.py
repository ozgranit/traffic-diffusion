from torchvision import transforms
import numpy as np
import cv2

MODELS_PATH = 'classification/classification_models/'

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