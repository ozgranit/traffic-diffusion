"""
The script demonstrates a simple example of using ART with PyTorch. The example train a small model on the MNIST dataset
and creates adversarial examples using the Fast Gradient Sign Method. Here we use the ART classifier to train the model,
it would also be possible to provide a pretrained model to the ART classifier.
The parameters are chosen for reduced computational requirements of the script and not optimised for accuracy.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

import matplotlib.pyplot as plt
from ShadowAttack.lisa import LisaCNN
from art.attacks.evasion import FastGradientMethod, ProjectedGradientDescent, CarliniL2Method, AutoAttack
from art.estimators.classification import PyTorchClassifier
from art.utils import load_cifar10, load_stl, load_iris
from art_attack import image_getter, LISA_NUM_OF_CLASSES
import torchvision.transforms as transforms
from shadow_attack_kaggle_images import attack as ortal_attack


class AdaptedModel(nn.Module):
    def __init__(self, original_model):
        super(AdaptedModel, self).__init__()

        # Resize the input images to 32x32
        self.resize = transforms.Resize((32, 32), antialias=True)

        # Load the pre-trained model
        self.original_model = original_model

    def forward(self, x):
        x = self.resize(x)  # Resize the input images
        x = self.original_model(x)  # Pass through the adapted model
        return x

def load_lisa_model():
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model = LisaCNN(n_class=LISA_NUM_OF_CLASSES).to(device)
    model.load_state_dict(torch.load(f'ShadowAttack/model/model_lisa.pth',
                                     map_location=torch.device(device)))
    model.eval()
    adapted_model = AdaptedModel(model)
    return adapted_model



# (x_train, y_train), (x_test, y_test), min_pixel_value, max_pixel_value = load_cifar10()
# # Step 1a: Swap axes to PyTorch's NCHW format
# x_train = np.transpose(x_train, (0, 3, 1, 2)).astype(np.float32)
# x_test = np.transpose(x_test, (0, 3, 1, 2)).astype(np.float32)
(x_train, y_train), (x_test, y_test), min_pixel_value, max_pixel_value = image_getter.load_cropped_larger_images()
assert len(x_test) == len(y_test)

# Step 2: Create the model

model = load_lisa_model()

# Step 2a: Define the loss function and the optimizer

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Step 3: Create the ART classifier

classifier = PyTorchClassifier(
    model=model,
    clip_values=(min_pixel_value, max_pixel_value),
    loss=criterion,
    optimizer=optimizer,
    input_shape=x_test[0].shape,
    nb_classes=LISA_NUM_OF_CLASSES,
)

# Step 4: Train the ART classifier

# classifier.fit(x_train, y_train, batch_size=64, nb_epochs=3)

# Step 5: Evaluate the ART classifier on benign test examples

predictions = classifier.predict(x_test)
accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
print("Accuracy on benign test examples: {}%".format(accuracy * 100))

# Step 6: Generate adversarial test examples
attack = ProjectedGradientDescent(estimator=classifier, eps=50, eps_step=1, num_random_init=5)
x_test_adv = []
done_idxes = set()
while len(x_test_adv) != len(x_test):
    print('full iteration')
    for idx, x in enumerate(x_test):
        if idx in done_idxes:
            continue
        x = np.expand_dims(x, axis=0)
        x_adv = attack.generate(x=x)
        if np.any(x-x_adv):
            print(f'succeeded on idx {idx}')
            x_test_adv += [x_adv]
            done_idxes.add(idx)

x_test_adv = np.concatenate(x_test_adv, axis=0)
# Step 7: Evaluate the ART classifier on adversarial test examples

predictions_adv = classifier.predict(x_test_adv)
accuracy = np.sum(np.argmax(predictions_adv, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
print("Accuracy on adversarial test examples: {}%".format(accuracy * 100))


def plot_images(images, labels, nrows, ncols):
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10, 10))
    for i, ax in enumerate(axes.flatten()):
        ax.imshow(images[i].astype(int))
        ax.set_title(f"Label: {np.argmax(labels[i])}")
        ax.axis('off')
    plt.tight_layout()
    plt.show()


x_test_adv_plt = np.transpose(x_test_adv, (0, 2, 3, 1))
x_test_plt = np.transpose(x_test, (0, 2, 3, 1))
plot_images(x_test_adv_plt, predictions_adv, nrows=3, ncols=5)
plot_images(x_test_plt, predictions, nrows=3, ncols=5)
plot_images(x_test_plt-x_test_adv_plt, predictions, nrows=3, ncols=5)


if __name__ == "__main__":
    pass
