import pandas as pd
import itertools
import numpy as np
import tensorflow as tf
from visualisation import show_mask_samples
from skimage.transform import resize
import cv2
from tqdm import tqdm
import memory_profiler
import gc
import h5py
#@memory_profiler.profile


def create_tf_dataset_from_hdf5(file_path, batch_size, chunk_size, train_ratio, train_shape):
    with h5py.File(file_path, 'r') as h5file:
        # Get the total number of samples
        total_samples = h5file['array1'].shape[0]
        train_samples = int(total_samples * train_ratio)
        test_samples = total_samples-train_samples

        # Create a generator for training data
        def train_data_generator():
            for i in range(0, train_samples, chunk_size):
                X_chunk = h5file['array1'][i:i + chunk_size]
                y_chunk = h5file['array2'][i:i + chunk_size]
                y_chunk = tf.keras.utils.to_categorical(y_chunk, num_classes=3)
                yield (X_chunk, y_chunk)

        # Create a generator for validation data
        def val_data_generator():
            for i in range(train_samples, total_samples, chunk_size):
                X_chunk = h5file['array1'][i:i + chunk_size]
                y_chunk = h5file['array2'][i:i + chunk_size]
                y_chunk = tf.keras.utils.to_categorical(y_chunk, num_classes=3)
                yield (X_chunk, y_chunk)

        # Create TensorFlow datasets
        train_dataset = tf.data.Dataset.from_generator(train_data_generator,
                                                       output_signature=(
                                                           tf.TensorSpec(shape=(None, train_shape[0], train_shape[1], 1),
                                                                         dtype=tf.float32),
                                                           tf.TensorSpec(shape=(None, train_shape[0], train_shape[1], 3),
                                                                         dtype=tf.float32)
                                                       ))

        val_dataset = tf.data.Dataset.from_generator(val_data_generator,
                                                     output_signature=(
                                                         tf.TensorSpec(shape=(None, train_shape[0], train_shape[1], 1),
                                                                       dtype=tf.float32),
                                                         tf.TensorSpec(shape=(None, train_shape[0], train_shape[1], 3),
                                                                       dtype=tf.float32)
                                                     ))

        # Shuffle and batch the datasets
        train_dataset = train_dataset.shuffle(buffer_size=1000).batch(batch_size).prefetch(
            buffer_size=tf.data.experimental.AUTOTUNE)
        val_dataset = val_dataset.batch(batch_size).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return train_dataset, val_dataset, train_samples, test_samples


def build_test_data(images, ev_image, im_size):
    if len(images) == 0 or len(ev_image) == 0:
        raise ValueError("Input lists must not be empty.")
    if len(ev_image) == 1:
        Xt1, Xt2 = [], []
        images = np.reshape(images, (len(images), im_size, im_size, 1))
        ev_image = np.reshape(ev_image, (len(ev_image), im_size, im_size, 1))

        return np.array(Xt1), np.array(Xt2)
    else:
        raise ValueError("Length of eval must be 1.")
