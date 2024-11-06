import os
import h5py
import numpy as np
import tensorflow as tf
import albumentations as A
from visualisation import plot_train_samples


def create_tf_dataset_from_hdf5(file_path, batch_size, chunk_size, train_ratio, train_shape, plot_samples):
    """

    :param file_path: str, path to processed dataset file
    :param batch_size: int
    :param chunk_size: int, large files should be loaded by chunks
    :param train_ratio: float, train/test ratio
    :param train_shape: tuple, target image shape
    :param plot_samples: boolean
    :return: tf.Dataset, tf.Dataset, int, int : train and validation datasets and their lengths
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError('train_dataset.hdf5 nor found')
    with h5py.File(file_path, 'r') as h5file:
        # Get the total number of samples
        total_samples = h5file['masks'].shape[0]
        train_samples = int(total_samples * train_ratio)
        test_samples = total_samples-train_samples
        prob = 0.5
        augmentation = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=prob),
            #A.RandomRotate90(p=prob),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, border_mode=0, p=prob),
            #A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=prob),
            #A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=prob),
            #A.GaussianBlur(blur_limit=(3, 5), p=prob),
            #A.GaussNoise(var_limit=(10.0, 50.0), p=prob),
            #A.ElasticTransform(alpha=1, sigma=50, p=prob),
            #A.GridDistortion(num_steps=5, distort_limit=0.3, p=prob),
            A.Resize(height=64, width=192),
        ])

        def train_data_generator():
            with h5py.File(file_path, 'r') as h5file:
                for i in range(0, train_samples, chunk_size):
                    X_chunk = h5file['traces_img'][i:i + chunk_size].astype(np.float32)
                    y_chunk = h5file['masks'][i:i + chunk_size].astype(np.float32)

                    for k in range(X_chunk.shape[0]):
                        augmented = augmentation(image=X_chunk[k], mask=y_chunk[k])
                        X_chunk[k] = augmented['image']
                        y_chunk[k] = augmented['mask']
                        del augmented
                    y_chunk = tf.keras.utils.to_categorical(y_chunk, num_classes=4)
                    for j in range(X_chunk.shape[0]):
                        yield X_chunk[j], y_chunk[j]
                    if plot_samples:
                        plot_train_samples(X_chunk[:20], y_chunk[:20], train_shape)

        def val_data_generator():
            with h5py.File(file_path, 'r') as h5file:
                for i in range(train_samples, total_samples, chunk_size):
                    X_chunk = h5file['traces_img'][i:i + chunk_size]
                    y_chunk = h5file['masks'][i:i + chunk_size]
                    y_chunk = tf.keras.utils.to_categorical(y_chunk, num_classes=3)
                    for j in range(X_chunk.shape[0]):
                        yield X_chunk[j], y_chunk[j]

        train_dataset = tf.data.Dataset.from_generator(train_data_generator,
                                                       output_signature=(
                                                           tf.TensorSpec(shape=(train_shape[0], train_shape[1], 1),
                                                                         dtype=tf.float32),
                                                           tf.TensorSpec(shape=(train_shape[0], train_shape[1], 4),
                                                                         dtype=tf.float32)
                                                       ))

        val_dataset = tf.data.Dataset.from_generator(val_data_generator,
                                                     output_signature=(
                                                         tf.TensorSpec(shape=(train_shape[0], train_shape[1], 1),
                                                                       dtype=tf.float32),
                                                         tf.TensorSpec(shape=(train_shape[0], train_shape[1], 4),
                                                                       dtype=tf.float32)
                                                     ))

        train_dataset = train_dataset.shuffle(buffer_size=500).batch(batch_size).repeat().prefetch(
            buffer_size=tf.data.experimental.AUTOTUNE)
        val_dataset = val_dataset.batch(batch_size).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return train_dataset, val_dataset, train_samples, test_samples

