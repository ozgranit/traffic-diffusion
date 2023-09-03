from pathlib import Path
from load_images import process_image
import numpy as np

import matplotlib.pyplot as plt
import torch
from ShadowAttack.lisa import LisaCNN
from art_attack import LISA_NUM_OF_CLASSES
from art_attack.my_attacks import PGDAttack
from shadow_attack_kaggle_images import load_generated_augmentations_by_type
from settings import GENERATED_IMAGES_TYPES_TRAIN, GENERATED_IMAGES_TYPES_TEST


def load_lisa_model():
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model = LisaCNN(n_class=LISA_NUM_OF_CLASSES).to(device)
    model.load_state_dict(torch.load(f'ShadowAttack/model/model_lisa.pth',
                                     map_location=torch.device(device)))
    model.eval()
    return model


traffic_diffusion_dir = Path.cwd()
# Load the image
image_dir = traffic_diffusion_dir / 'larger_images'

file_names, _, _, cropped_resized_imgs, _, bbx, _ = process_image(str(image_dir / 'image_inputs'),
                                                                  str(image_dir / 'image_annotations'),
                                                                  'LISA', crop_size=32,
                                                                  mask_folder=str(image_dir / 'image_masks'))

# Swap axes to PyTorch's NCHW format
input_images = [np.transpose(image, (2, 0, 1)) for image in cropped_resized_imgs]
input_images = torch.tensor(np.stack(input_images, axis=0)) / 255
model = load_lisa_model()
labels = torch.tensor([12]*len(cropped_resized_imgs))

pgd_attack = PGDAttack(model=model, eps=50/255, n=100)
x_adv = pgd_attack.execute(input_images, labels)


def plot_images(images, labels, nrows, ncols):
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10, 10))
    for i, ax in enumerate(axes.flatten()):
        ax.imshow(images[i])
        ax.set_title(f"Label: {np.argmax(labels[i])}")
        ax.axis('off')
    plt.tight_layout()
    plt.show()


orig_predictions = model(input_images).detach().numpy()
accuracy = np.sum(np.argmax(orig_predictions, axis=1) == 12) / len(labels)
print("Accuracy on benign test examples: {}%".format(accuracy * 100))


adv_predictions = model(x_adv).detach().numpy()
accuracy = np.sum(np.argmax(adv_predictions, axis=1) == 12) / len(labels)
print("Accuracy on adversarial test examples: {}%".format(accuracy * 100))

plot_images(cropped_resized_imgs, [12] * 26, nrows=3, ncols=5)


def imshow_batch(tensor):
    # Convert tensor to a NumPy array
    images = tensor.detach().numpy()

    # Change the shape from (batch, channels, height, width) to (batch, height, width, channels)
    images = np.transpose(images, (0, 2, 3, 1))

    return images


plot_images(imshow_batch(x_adv), [12] * 26, nrows=3, ncols=5)


if __name__ == "__main__":
    pass
