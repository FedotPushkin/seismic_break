import os
import gc
import numpy as np
import tensorflow as tf
from tensorflow import keras
from visualisation import plothistory
from sklearn.utils.class_weight import compute_class_weight
from visualisation import show_performance_metrics, make_predictions
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras_unet_collection.models import unet_3plus_2d
from sklearn.model_selection import train_test_split
from tensorflow_addons.losses import SigmoidFocalCrossEntropy

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
                             filter_num_down=[32, 64, 128],
                             filter_num_skip='auto',
                             filter_num_aggregate=96,
                             stack_num_down=2,
                             stack_num_up=1,
                             activation='ReLU',
                             output_activation='Softmax',
                             batch_norm=True,
                             pool='max',
                             unpool='bilinear',
                             deep_supervision=True,
                             n_labels=1,)

        def dice_coefficient(y_true, y_pred, smooth=1):
            intersection = tf.reduce_sum(y_true * y_pred)
            return (2. * intersection + smooth) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + smooth)

        model.compile(optimizer='adam',
                      loss=SigmoidFocalCrossEntropy(),
                      metrics=dice_coefficient)

        return model

    def fit_to_data(self, X, y, batch_size, epochs, show_perf):

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        with tf.device('/GPU:0'):
            gc.collect()

            if not os.path.exists('models'):
                os.makedirs('models')

            early_stop = keras.callbacks.EarlyStopping(monitor='val_loss',
                                                       patience=5,
                                                       verbose=1,
                                                       restore_best_weights=True
                                                       )
            dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
            dataset = dataset.repeat()  # Repeat for indefinite training
            dataset = dataset.batch(batch_size).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
            val_set = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(batch_size).\
                batch(batch_size).prefetch(buffer_size=tf.data.experimental.AUTOTUNE).repeat()

            self.history = self.model.fit(dataset,
                                          validation_data=val_set,
                                          epochs=epochs,
                                          steps_per_epoch=y_train.shape[0] // batch_size,
                                          validation_steps=y_test.shape[0] // batch_size,)
                                          #callbacks=[early_stop])

            self.model.save(f'models/Unet3Plus_model.h5')

            gc.collect()
        if self.history is not None:
            if show_perf:
                plothistory(self.history.history,)
                y_pred = make_predictions(X_test)
                show_performance_metrics(self.y_test, y_pred)

