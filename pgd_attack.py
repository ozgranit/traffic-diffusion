import torch
import numpy as np
import matplotlib.pyplot as plt

from ShadowAttack.lisa import LisaCNN
from art_attack import LISA_NUM_OF_CLASSES
from art_attack.my_attacks import PGDAttack
from art_attack.image_getter import image_generator


def plot_images(images, labels, nrows, ncols):
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10, 10))
    for i, ax in enumerate(axes.flatten()):
        ax.imshow(images[i])
        ax.set_title(f"Label: {np.argmax(labels[i])}")
        ax.axis('off')
    plt.tight_layout()
    plt.show()


def imshow_batch(images: list | torch.Tensor):
    if isinstance(images, list):
        images = torch.concat(images, dim=0)
    # Convert tensor to a NumPy array
    images = images.detach().numpy()

    # Change the shape from (batch, channels, height, width) to (batch, height, width, channels)
    images = np.transpose(images, (0, 2, 3, 1))

    return images


def load_lisa_model():
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model = LisaCNN(n_class=LISA_NUM_OF_CLASSES).to(device)
    model.load_state_dict(torch.load(f'ShadowAttack/model/model_lisa.pth',
                                     map_location=torch.device(device)))
    model.eval()
    return model


def main():
    model = load_lisa_model()
    images_for_plot, labels_for_plot, adv_images_for_plot, adv_labels_for_plot = [], [], [], []
    orig_accuracy, adv_accuracy, count = 0, 0, 0
    test_accuracy, attacked_test_accuracy = 0, 0

    for orig_image, train_images, test_images in image_generator():

        # Swap axes to PyTorch's NCHW format
        input_image = [np.transpose(orig_image, (2, 0, 1))]
        input_image = torch.tensor(np.stack(input_image, axis=0)) / 255
        label = torch.tensor([12])

        pgd_attack = PGDAttack(model=model, eps=40/255, n=100)
        x_adv = pgd_attack.execute(input_image, label)

        orig_predictions = model(input_image).detach().numpy()
        orig_accuracy += np.sum(np.argmax(orig_predictions, axis=1) == label.item())
        images_for_plot.append(input_image)
        labels_for_plot.append(orig_predictions)

        adv_predictions = model(x_adv).detach().numpy()
        adv_accuracy += np.sum(np.argmax(adv_predictions, axis=1) == label.item())
        adv_images_for_plot.append(x_adv)
        adv_labels_for_plot.append(adv_predictions)
        count += 1

        # apply attack to test images
        attack_only = x_adv.detach().numpy() - input_image.detach().numpy()
        for test_image in test_images:
            test_image = [np.transpose(test_image, (2, 0, 1))]
            test_image = torch.tensor(np.stack(test_image, axis=0)) / 255

            test_pred = model(test_image).detach().numpy()
            test_accuracy += np.sum(np.argmax(test_pred, axis=1) == label.item()) / len(test_images)

            attack_pred = model(test_image + attack_only).detach().numpy()
            attacked_test_accuracy += np.sum(np.argmax(attack_pred, axis=1) == label.item()) / len(test_images)

    print(f'Accuracy on benign examples: {orig_accuracy * 100 / count}%')
    print(f'Accuracy on adversarial examples: {adv_accuracy * 100 / count}%')

    print(f'Accuracy on test images: {test_accuracy * 100 / count}%')
    print(f'Accuracy on attacked test images: {attacked_test_accuracy * 100 / count}%')

    plot_images(imshow_batch(images_for_plot), labels_for_plot, nrows=5, ncols=5)
    plot_images(imshow_batch(adv_images_for_plot), adv_labels_for_plot, nrows=5, ncols=5)


if __name__ == "__main__":
    main()
