import numpy as np
import matplotlib.pyplot as plt
from utils import rgb_to_grayscale


def plothistory(history):

    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    try:
        if not isinstance(history, dict):
            raise TypeError("Expected a dictionary for history")
        plt.plot(history['unet3plus_output_final_activation_loss'], label='loss')
        plt.plot(history['val_unet3plus_output_final_activation_loss'], label='val_loss')
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(loc='upper left')

        plt.subplot(2, 1, 2)
        plt.plot(history['val_unet3plus_output_final_activation_precision'], label='_precision')
        plt.plot(history['val_unet3plus_output_final_activation_recall'], label='val_recall')
        plt.plot(history['val_unet3plus_output_sup0_activation_iou_score'], label='val_iou')
        plt.plot(history['unet3plus_output_final_activation_precision'], label='precision')
        plt.plot(history['unet3plus_output_final_activation_recall'], label='recall')
        plt.plot(history['unet3plus_output_final_activation_iou_score'], label='iou')

        plt.title('Model Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(loc='upper left')

        plt.tight_layout()
        plt.show(block=True)
    except KeyError as e:
        print(f"KeyError: {e} - Check if the required keys are in the history dictionary.")


def show_mask_samples(images):
    if images is None or images.any() is None:
        raise ValueError('One of arguments is None')
    if len(images) == 0:
        raise ValueError('empty argument recieved')
    images = images[:6]
    fig, axes = plt.subplots(3, 2, figsize=(10, 7))

    for idx, ax in enumerate(axes.flat):
        ax.imshow(images[idx])
        ax.axis('off')
    plt.tight_layout()
    plt.show(block=True)


def plot_train_samples(arrayX, arrayY, train_shape):
    num_plots = 4
    fig, axs = plt.subplots(num_plots, 2, figsize=(10, num_plots * 3))
    arrayX = np.squeeze(arrayX, axis=-1)
    arrayY = np.squeeze(arrayY, axis=-1)
    bias = 0
    # Loop through the arrays and plot them
    for i in range(num_plots):
        y_indices, x_indices = np.where(arrayY[i+bias] == 1)
        axs[i, 0].imshow(arrayX[i+bias], cmap='seismic', aspect='auto')  # Left column
        axs[i, 0].set_title(f'Train samples Array X {i + 1}')
        axs[i, 0].axis('off')  # Hide axes
        if len(x_indices) > 0:
            axs[i, 0].plot(x_indices, y_indices, color='black', markersize=2)

        axs[i, 1].imshow(arrayY[i+bias], cmap='seismic', aspect='auto')  # Right column
        axs[i, 1].set_title(f'Array Y {i + 1}')
        axs[i, 1].axis('off')  # Hide axes

    # Adjust layout
    plt.tight_layout()
    plt.show()


def plot_train_sample(arrayX, arrayY):

    plt.subplots(1, figsize=(10, 10))
    plt.imshow(arrayX, cmap='seismic', aspect='auto')
    plt.title(f'Train sample Array X ')

    x_indices = np.linspace(0, len(arrayY), len(arrayY))
    plt.plot(arrayY.flatten(), x_indices, color='black', markersize=2, linestyle='dotted')

    plt.tight_layout()
    plt.show(block=True)


def show_predicted_images(y_test, y_pred):

    if y_test is None or y_pred is None:
        raise ValueError('one of arguments is None')
    fig, axes = plt.subplots(8, 2, figsize=(10, 20))

    axes[0, 0].set_title('Truth')
    axes[0, 1].set_title('Prediction')
    for i in range(8):

        grayscale_image = rgb_to_grayscale(y_test[i])
        # Plot ground truth on the left column
        axes[i, 0].imshow(grayscale_image, cmap='seismic')
        axes[i, 0].axis('off')  # Hide axes
        # Plot prediction on the right column
        grayscale_image = rgb_to_grayscale(y_pred[i])
        axes[i, 1].imshow(grayscale_image, cmap='seismic')
        axes[i, 1].axis('off')  # Hide axes

    plt.tight_layout()
    plt.show(block=True)




