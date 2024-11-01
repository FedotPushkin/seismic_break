import os
import math
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

def plothistory(history):

    plt.figure(figsize=(12, 6))

    plt.subplot(2, 1, 1)
    plt.plot(history['loss'], label='train_loss')
    plt.plot(history['val_loss'], label='val_loss')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(loc='upper left')

    plt.subplot(2, 1, 2)
    if 'accuracy' not in history or 'val_accuracy' not in history:
        raise KeyError("History object missing 'accuracy' or 'val_accuracy' keys.")
    plt.plot(history['accuracy'], label='train_acc')
    plt.plot(history['val_accuracy'], label='val_acc')
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(loc='upper left')

    plt.tight_layout()
    plt.show(block=True)


def show_mask_samples(images, names):
    if len(images) > 6:
       images = images[:6]
    if len(images) > 2:
        height = 2
        width = len(images)//2
    else:
        height = 1
        width = len(images)

    fig, axes = plt.subplots(height, width, figsize=(10, 7))
    if images is None or names is None:
        raise ValueError('one of arguments is None')
    if len(images) == 0 or len(names) == 0:
        raise ValueError('empty argument recieved')
    for idx, ax in enumerate(axes.flat):
        ax.imshow(images[idx])
        ax.set_title(names[idx])
        ax.axis('off')
    plt.tight_layout()
    plt.show(block=True)


def plot_train_samples(arrayX, arrayY,train_shape):
    num_plots = 3
    fig, axs = plt.subplots(num_plots, 2, figsize=(10, num_plots * 3))
    arrayX = np.reshape(arrayX, (len(arrayX), train_shape[0], train_shape[1]))
    arrayY = np.reshape(arrayY, (len(arrayY), train_shape[0], train_shape[1]))
    bias = 3
    # Loop through the arrays and plot them
    for i in range(num_plots):
        y_indices, x_indices = np.where(arrayY[i+bias] == 1)
        axs[i, 0].imshow(arrayX[i+bias], cmap='seismic', aspect='auto')  # Left column
        axs[i, 0].set_title(f'Array X {i + 1}')
        axs[i, 0].axis('off')  # Hide axes
        if len(x_indices) > 0:
            axs[i, 0].plot(x_indices, y_indices, 'ro', markersize=2)

        axs[i, 1].imshow(arrayY[i+bias], cmap='seismic', aspect='auto')  # Right column
        axs[i, 1].set_title(f'Array Y {i + 1}')
        axs[i, 1].axis('off')  # Hide axes

    # Adjust layout
    plt.tight_layout()
    plt.show(block=True)


def plot_train_sample(arrayX, arrayY):

    plt.subplots(1, figsize=(10, 10))

    bias = 0
    plt.imshow(arrayX, cmap='seismic', aspect='auto')  # Left column
    plt.title(f'Array X ')

    x_indices = np.linspace(0, len(arrayY), len(arrayY))
    plt.plot( arrayY.flatten(),x_indices, 'ro', markersize=2)

    #plt.imshow(arrayY, cmap='seismic', aspect='auto')  # Right column
    #plt.title(f'Array Y ')

    # Adjust layout
    plt.tight_layout()
    plt.show(block=True)


def make_predictions(X_test):
    res = []
    for i in range(1):
        model_name = f'Unet3Plus_model.h5'
        model_path = os.path.join('models/', model_name)
        if os.path.exists(model_path):
            model = load_model(model_path)
        else:
            raise FileNotFoundError(f"Model file {model_name} not found.")

        y_pred = model.evaluate(X_test, batch_size=32)
        y_pred = np.resize(y_pred, len(y_pred))
        res.append(y_pred)
    return np.mean(res)


def show_performance_metrics(y, y_pred):

    if y is None or y_pred is None:
        raise ValueError('one of arguments is None')

    else:
        raise ValueError('empty argument recieved')


def show2withaxis(a, b):
    plt.figure(figsize=(10, 5))
    xa = np.linspace(0, len(a), len(a))
    xb = np.linspace(0, len(b), len(b))
    # Plotting the first array
    plt.plot(xa, a, label='orig', color='b')

    # Plotting the second array
    plt.plot(xb, b, label='resized', color='r')

    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')

    # Adding legend
    plt.legend()

    # Show grid
    plt.grid()

    # Show the plot
    plt.show(block = True)
