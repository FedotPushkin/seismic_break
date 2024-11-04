import os
import h5py
import numpy as np
import tensorflow as tf
import albumentations as augm
from visualisation import plot_train_samples


def create_tf_dataset_from_hdf5(file_path, batch_size, chunk_size, train_ratio, train_shape,plot_samples):
    if not os.path.isfile(file_path):
        raise FileNotFoundError('train_dataset.hdf5 nor found')
    with h5py.File(file_path, 'r') as h5file:
        # Get the total number of samples
        total_samples = h5file['masks'].shape[0]
        train_samples = int(total_samples * train_ratio)
        test_samples = total_samples-train_samples
        augmentation = augm.Compose([
            augm.HorizontalFlip(p=0.5),
            #   augm.VerticalFlip(p=0.1),
            #   augm.RandomRotate90(p=0.1),
            #   augm.RandomBrightnessContrast(p=0.2),
            augm.Resize(height=64, width=192),  # Resize to your target shape
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
                    y_chunk = tf.keras.utils.to_categorical(y_chunk, num_classes=3)
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
                                                           tf.TensorSpec(shape=(train_shape[0], train_shape[1], 3),
                                                                         dtype=tf.float32)
                                                       ))

        val_dataset = tf.data.Dataset.from_generator(val_data_generator,
                                                     output_signature=(
                                                         tf.TensorSpec(shape=(train_shape[0], train_shape[1], 1),
                                                                       dtype=tf.float32),
                                                         tf.TensorSpec(shape=(train_shape[0], train_shape[1], 3),
                                                                       dtype=tf.float32)
                                                     ))

        train_dataset = train_dataset.shuffle(buffer_size=100).batch(batch_size).repeat().prefetch(
            buffer_size=100)
        val_dataset = val_dataset.batch(batch_size).prefetch(buffer_size=100)#tf.data.experimental.AUTOTUNE)

    return train_dataset, val_dataset, train_samples, test_samples

