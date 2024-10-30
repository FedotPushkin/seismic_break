import os
import gc
import numpy as np
import tensorflow as tf
from tensorflow import keras
from visualisation import plothistory
from tensorflow.keras import layers, models, regularizers
from sklearn.model_selection import KFold
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout,  GlobalAveragePooling2D
from sklearn.utils.class_weight import compute_class_weight
from visualisation import show_performance_metrics, make_predictions
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
import segmentation_models as sm
from keras_unet_collection.models import unet_3plus_2d
from sklearn.model_selection import train_test_split

class Unet_NN:

    def __init__(self, input_shape):

        self.x1_train, self.x1_test, self.y_train, self.y_test = \
            None, None, None, None
        self.X1 = None
        self.y = None
        self.im_width = input_shape[0]
        self.im_height = input_shape[1]
        self.history = None
        self.model = self.setup_unet(input_shape)
    @staticmethod
    def setup_unet(input_shape):

        model = unet_3plus_2d(
                             input_size=input_shape,
                             filter_num_down=[64, 128, 256, 512, 1024],
                             filter_num_skip='auto',
                             filter_num_aggregate=160,
                             stack_num_down=2,
                             stack_num_up=1,
                             activation='ReLU',
                             output_activation='Softmax',
                             batch_norm=True,
                             pool='max',
                             unpool='bilinear',
                             deep_supervision=True,
                             n_labels=3,)



        #for layer in base_network.layers:
        #    layer.trainable = False
        #for layer in base_network.layers[-3:]:
        #    layer.trainable = True
        return model

    def fit_to_data(self, X, y, batch_size, epochs, show_perf):

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        with tf.device('/GPU:0'):
            gc.collect()

            if not os.path.exists('models'):
                os.makedirs('models')

            dataset, val_set = self.get_datasets(X_train, y_train, X_test, y_test, batch_size)
            early_stop = keras.callbacks.EarlyStopping(monitor='val_loss',
                                                       patience=5,
                                                       verbose=1,
                                                       restore_best_weights=True
                                                       )
            self.history = self.model.fit(dataset,
                                          validation_data=val_set,
                                          epochs=epochs,
                                          steps_per_epoch=len(y_train) // batch_size,
                                          validation_steps=len(y_test) // batch_size,
                                          callbacks=[early_stop])

            self.model.save(f'models/Unet3Plus_model.h5')

            gc.collect()
        if self.history is not None:
            if show_perf:
                plothistory(self.history.history)
                y_pred = make_predictions(X_test)
                show_performance_metrics(self.y_test, y_pred)

    def custom_generator(self, X1, y, batch_size, train_generator):
        num_samples = X1.shape[0]
        while True:
            indices = np.random.permutation(num_samples)  # Shuffle indices
            for start in range(0, num_samples, batch_size):
                end = min(start + batch_size, num_samples)
                batch_indices = indices[start:end]

                # Apply augmentation to both input images
                X1_augmented = train_generator.flow(X1[batch_indices], batch_size=batch_size, shuffle=False)

                # Get the augmented images
                X1_batch = next(X1_augmented)


                # Yield the two augmented input images and their labels
                yield X1_batch, y[batch_indices]

    def get_datasets(self,X1_t, X2_t, y_t, X1_v, X2_v, y_v, batch_size):

        im_gen = ImageDataGenerator(
            rotation_range=10,
            width_shift_range=0.05,
            height_shift_range=0.05,
            shear_range=0.01,
            zoom_range=[0.95, 1.05],

            fill_mode='nearest'
        )
        train_gen = self.custom_generator(X1_t, y_t, batch_size, im_gen)
        val_gen = self.custom_generator(X1_v,  y_v, batch_size, ImageDataGenerator())
        signature = (tf.TensorSpec(shape=(None, self.im_size, self.im_size, 2), dtype=tf.float16),
                     tf.TensorSpec(shape=(None,), dtype=tf.float16))

        train_dataset = tf.data.Dataset.from_generator(lambda: train_gen, output_signature=signature). \
            prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

        validation_set = tf.data.Dataset.from_generator(lambda: val_gen, output_signature=signature). \
            prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        return train_dataset, validation_set
