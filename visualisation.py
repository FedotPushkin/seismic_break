import os
import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import auc
from sklearn.metrics import roc_curve
from sklearn.metrics import f1_score, precision_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
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


def plotauc(y_pred_p, yval_p, mess):
    fpr, tpr, thresholds = roc_curve(yval_p, y_pred_p)
    auc_siam = auc(fpr, tpr)

    plt.figure(figsize=(12, 6))
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr, tpr, label='Siamese network (area = {:.3f})'.format(auc_siam))
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title(f'ROC curve {mess}')
    plt.legend(loc='best')
    plt.grid(True)
    plt.show()


def show_image_samples(images, names):
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


def plot_df_samples(df):
    num_images = 3
    fig, axes = plt.subplots(1, num_images, figsize=(15, 5))
    for i in range(num_images):
        bias = 0
        # Display the image
        axes[i].imshow(df['2d_array'][i+bias].T, cmap='gray')

        # Plot the corresponding line
        axes[i].plot(df['1d_array'][i+bias], color='red', linewidth=1)

        # Set title or labels if needed
        axes[i].set_title(f'Image {i+bias + 1}')
        axes[i].axis('off')  # Hide axes ticks

    # Adjust layout
    plt.tight_layout()
    plt.show(block=True)


def make_predictions(X1, X2):
    res = []
    for i in range(1):
        model_name = f'my_model_f16_k5_l1_fold{i}.h5'
        model_path = os.path.join('models/', model_name)
        if os.path.exists(model_path):
            model = load_model(model_path)
        else:
            raise FileNotFoundError(f"Model file {model_name} not found.")

        y_pred = model.predict([X1, X2], batch_size=32)
        y_pred = np.resize(y_pred, len(y_pred))
        res.append(y_pred)
    return np.mean(res)


def show_performance_metrics(y, y_pred):

    if y is None or y_pred is None:
        raise ValueError('one of arguments is None')
    if y.size > 0 and y_pred.size > 0:
        y_pred = (y_pred >= 0.5).astype(int)
        plotauc(y_pred, y, 'auc')

        cm = confusion_matrix(y, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap=plt.cm.Blues)  # Optional: change the color map
        plt.title("Confusion Matrix")
        plt.show(block=True)
        plt.savefig('confusion_matrix.png')
        plt.close()

        precision = precision_score(y, y_pred)
        f1 = f1_score(y, y_pred)

        print(f"F1 Score: {f1}")
        print(f"Precision Score: {precision}")
    else:
        raise ValueError('empty argument recieved')
