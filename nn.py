import os
import gc
import json
import tensorflow as tf
from tensorflow import keras
from visualisation import plothistory,plot_batch_eposh_hist
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import Callback
from keras_unet_collection.models import unet_3plus_2d
from segmentation_models.metrics import IOUScore, Precision, Recall
from segmentation_models.losses import DiceLoss, CategoricalFocalLoss


class Unet_NN:
    """
    Class implements initialisation, setup, fitting , evaluation and predicting
    """

    def __init__(self, input_shape):

        self.im_width = input_shape[0]
        self.im_height = input_shape[1]
        self.history = None
        self.model = self.setup_unet(input_shape)

    @staticmethod
    def dice_loss_plus_focal_loss(y_true, y_pred):
        return DiceLoss()(y_true, y_pred) + CategoricalFocalLoss(gamma=2.0)(y_true, y_pred)

    def setup_unet(self, input_shape):

        model = unet_3plus_2d(
                             input_size=input_shape,
                             filter_num_down=[32, 64, 128],
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
                             n_labels=4,)

        loss_fn = self.dice_loss_plus_focal_loss
        metrics = [IOUScore(class_indexes=[0, 1, 2, 3]), Precision(class_indexes=[1]), Recall(class_indexes=[1])]
        model.compile(optimizer='adam',
                      loss=loss_fn,
                      metrics=metrics)

        return model

    def fit_to_data(self, dataset, val_dataset, batch_size, epochs, show_perf, train_samples, test_samples):
        '''

        :param dataset: tf.Dataset
        :param val_dataset: f.Dataset
        :param batch_size: int
        :param epochs: int
        :param show_perf: boolean
        :param train_samples: int
        :param test_samples: int
        :return: None
        '''

        with tf.device('/GPU:0'):
            gc.collect()

            if not os.path.exists('models'):
                os.makedirs('models')

            early_stop = keras.callbacks.EarlyStopping(monitor='val_unet3plus_output_final_activation_loss',
                                                       patience=3,
                                                       verbose=1,
                                                       restore_best_weights=True
                                                       )
            loss_history_callback = LossHistoryCallback()
            self.history = self.model.fit(dataset,
                                          validation_data=val_dataset,
                                          epochs=epochs,
                                          verbose=0,
                                          steps_per_epoch=train_samples // (batch_size),
                                          validation_steps=test_samples // (batch_size),
                                          callbacks=[early_stop, CustomMetricsLogger(), loss_history_callback])

            self.model.save(f'models/Unet3Plus_model.h5')
            gc.collect()

            plot_batch_eposh_hist(loss_history_callback)
        if self.history is not None:
            with open('training_history.json', 'w') as file:
                json.dump(self.history.history, file)
            if show_perf:
                plothistory(self.history.history)

    def make_predictions(self, val_dataset, batch_size, test_samples):

        model_name = f'Unet3Plus_model.h5'
        model_path = os.path.join('models/', model_name)
        if os.path.exists(model_path):
            model = load_model(model_path, custom_objects={
                'dice_loss_plus_focal_loss': self.dice_loss_plus_focal_loss,
                'iou_score': IOUScore(class_indexes=[0, 1, 2, 3]),
                'precision': Precision(class_indexes=[1]),
                'recall': Recall(class_indexes=[1])
            })
        else:
            raise FileNotFoundError(f"Model file {model_name} not found.")

        X_test = []
        y_test = []
        num_batches = 1
        for x_batch, y_batch in val_dataset.take(num_batches):
            X_test.append(x_batch.numpy())
            y_test.append(y_batch.numpy())

        prediction = model.predict(X_test[0])
        evaluation_results = model.evaluate(val_dataset, steps=test_samples // batch_size, return_dict=True)
        loss = evaluation_results['loss']
        print(f'loss {loss}')
        for metric_name, metric_value in evaluation_results.items():
            print(f"{metric_name}: {metric_value}")
        return y_test[0], prediction[0]


class CustomMetricsLogger(Callback):
    """
    Metrics and losses have long names so this code shrinks them to  fit on the screen
    """
    @staticmethod
    def on_batch_end(batch, logs=None):
        if 'unet3plus_output_final_activation_loss' in logs \
            and 'unet3plus_output_final_activation_iou_score' in logs \
                and 'unet3plus_output_final_activation_precision' in logs \
                and 'unet3plus_output_final_activation_recall' in logs:
            print(f"Batch {batch + 1}  Loss = {logs['unet3plus_output_final_activation_loss']:.4f}"
                  f" Precicion = {logs['unet3plus_output_final_activation_precision']:.4f}"
                  f" IoU = {logs['unet3plus_output_final_activation_iou_score']:.4f}"
                  f" Recall = {logs['unet3plus_output_final_activation_recall']:.4f}")


class LossHistoryCallback(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        # Initialize lists to store losses and metrics for each epoch and batch
        self.epoch_loss = []
        self.epoch_metrics = []
        self.batch_loss = []
        self.batch_metrics = []
        self.epochs = []
        self.batches = []

    def on_epoch_end(self, epoch, logs=None):
        # Append loss and metrics after each epoch
        self.epoch_loss.append(logs.get('loss'))
        self.epoch_metrics.append(logs.get('unet3plus_output_final_activation_iou_score'))
        self.epochs.append(epoch)

    def on_batch_end(self, batch, logs=None):
        # Append loss and metrics after each batch
        self.batch_loss.append(logs.get('loss'))
        self.batch_metrics.append(logs.get('unet3plus_output_final_activation_iou_score'))
        self.batches.append(batch)
