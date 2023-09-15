import numpy as np
from PIL import Image
from pgd_attack import LISA_NUM_OF_CLASSES
import matplotlib.pyplot as plt
from pathlib import Path
from shadow_attack_kaggle_images import load_generated_augmentations_by_type
from settings import GENERATED_IMAGES_TYPES_TRAIN, GENERATED_IMAGES_TYPES_TEST
from load_images import process_image


def load_larger_images():
    traffic_diffusion_dir = Path.cwd()
    # Load the image
    image_dir = traffic_diffusion_dir / 'larger_images' / 'image_inputs'

    # List of image paths
    image_paths = [image_dir / f'road_{i}.jpg' for i in range(1, 27)]

    # Load and reshape all images
    reshaped_images = [load_and_reshape_image(image_path) for image_path in image_paths]

    # Combine the reshaped images into a single NumPy array
    all_images = np.concatenate(reshaped_images, axis=0)

    min_pixel_value = 0
    max_pixel_value = 255
    x_test = all_images
    y_test = get_stop_lisa_labels(len(all_images))
    x_train = None
    y_train = None

    return (x_train, y_train), (x_test, y_test), min_pixel_value, max_pixel_value


def get_stop_lisa_labels(num_duplicates: int = 1) -> np.ndarray:
    stop_label_loc = 12
    label = [0] * LISA_NUM_OF_CLASSES
    label[stop_label_loc] = 1

    # Convert the list to a NumPy ndarray and repeat it
    duplicated_list = [label for _ in range(num_duplicates)]

    # Reshape the ndarray to have the desired number of rows
    return np.array(duplicated_list)


def load_and_reshape_image(image_path, target_size=(224, 224)):
    # Load the image
    image = Image.open(image_path)

    # Resize the image
    resized_image = image.resize(target_size, Image.ANTIALIAS)

    # Convert the image to a NumPy ndarray
    image_np = np.array(resized_image)

    # Swap axes to PyTorch's NCHW format
    image_np = np.transpose(image_np, (2, 0, 1))
    image_np = np.expand_dims(image_np, axis=0)  # Adding batch dimension

    return image_np


def show_images(image_paths):
    # Load, reshape, and display all images
    plt.figure(figsize=(10, 5))

    for i, image_path in enumerate(image_paths):
        resized_image, _ = load_and_reshape_image(image_path)

        # Display the resized image
        plt.subplot(1, len(image_paths), i + 1)
        plt.imshow(resized_image)
        plt.title(f"Image {i + 1}")
        plt.axis("off")

    plt.show()


def load_cropped_larger_images():
    traffic_diffusion_dir = Path.cwd()
    # Load the image
    image_dir = traffic_diffusion_dir / 'larger_images'

    _, _, _, cropped_resized_imgs, _, _, _ = process_image(str(image_dir / 'image_inputs'),
                                                           str(image_dir / 'image_annotations'),
                                                           'LISA', crop_size=224,
                                                           mask_folder=str(image_dir / 'image_masks'))

    # Swap axes to PyTorch's NCHW format
    cropped_resized_imgs = [np.transpose(image, (2, 0, 1)) for image in cropped_resized_imgs]
    # cropped_resized_imgs = [image / 255 for image in cropped_resized_imgs]
    # Combine the reshaped images into a single NumPy array
    all_images = np.stack(cropped_resized_imgs, axis=0)
    all_labels = get_stop_lisa_labels(len(all_images))

    min_pixel_value = 0
    max_pixel_value = 255
    x_test = all_images
    y_test = all_labels
    x_train = None
    y_train = None
    # (26, 3, 32, 32)
    return (x_train, y_train), (x_test, y_test), min_pixel_value, max_pixel_value


def image_generator(size: int = 32):
    """returns image, test generated images and train generated images"""

    def generate_train(image_dir):
        return load_generated_augmentations_by_type(GENERATED_IMAGES_TYPES_TRAIN, image_dir, bbx[i], to_size=size)

    def generate_test(image_dir):
        return load_generated_augmentations_by_type(GENERATED_IMAGES_TYPES_TEST, image_dir, bbx[i], to_size=size)

    traffic_diffusion_dir = Path.cwd()
    assert traffic_diffusion_dir.name == 'traffic-diffusion', 'got the wrong folder'
    larger_images_dir = traffic_diffusion_dir / 'larger_images'

    file_names, _, _, cropped_images, _, bbx, _ = process_image(str(larger_images_dir / 'image_inputs'),
                                                                str(larger_images_dir / 'image_annotations'),
                                                                'LISA', crop_size=size,
                                                                mask_folder=str(larger_images_dir / 'image_masks'))
    for i, image_name in enumerate(file_names):
        original_image = cropped_images[i]

        diffusion_dir = larger_images_dir / 'image_outputs' / image_name
        generated_train_images, _ = generate_train(diffusion_dir)
        generated_test_images, _ = generate_test(diffusion_dir)

        yield original_image, generated_train_images, generated_test_images


if __name__ == "__main__":
    pass
