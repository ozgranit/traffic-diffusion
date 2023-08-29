from matplotlib import pyplot as plt


def plot_2_images_in_a_row(image1, image2, title1: str="", title2: str="", cmap=None, figsize=(10, 5), save_path=None, plot=True):
    # Create a figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Plot the first image on the first subplot
    axes[0].imshow(image1, cmap=cmap)  # Use cmap='gray' for grayscale images_orig
    axes[0].set_title(title1)

    # Plot the second image on the second subplot
    axes[1].imshow(image2, cmap=cmap)  # Use cmap='gray' for grayscale images_orig
    axes[1].set_title(title2)

    # Adjust layout to prevent overlapping titles and labels
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)  # Save the plot as an image
        print(f"Plot saved as {save_path}")
    if plot:
        plt.show()