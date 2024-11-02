import os
import gc
import numpy as np
import tensorflow as tf
from tensorflow import keras
from visualisation import plothistory
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.metrics import MeanIoU
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras_unet_collection.models import unet_3plus_2d
from sklearn.model_selection import train_test_split
from tensorflow_addons.losses import SigmoidFocalCrossEntropy
from segmentation_models.metrics import IOUScore, Precision, Recall
from segmentation_models.losses import DiceLoss, JaccardLoss, CategoricalFocalLoss
import json
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import Callback

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
                             n_labels=3,)

        loss_fn = self.dice_loss_plus_focal_loss
        metrics = [IOUScore(class_indexes=[0, 1, 2]), Precision(class_indexes=[1]), Recall(class_indexes=[1])]
        model.compile(optimizer='adam',
                      loss=loss_fn,
                      metrics=metrics)

        return model

    def fit_to_data(self, dataset, val_dataset, batch_size, epochs, show_perf, train_samples, test_samples):

       # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        with tf.device('/GPU:0'):
            gc.collect()

            if not os.path.exists('models'):
                os.makedirs('models')

            early_stop = keras.callbacks.EarlyStopping(monitor='val_unet3plus_output_final_activation_loss',
                                                       patience=5,
                                                       verbose=1,
                                                       restore_best_weights=True
                                                       )

            #y_train = tf.keras.utils.to_categorical(y_train, num_classes=3)
            #y_test = tf.keras.utils.to_categorical(y_test, num_classes=3)
            #dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
            #dataset = dataset .shuffle(buffer_size=100).batch(batch_size).repeat().prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
            #val_set = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(batch_size).repeat().\
            #    prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
            class CustomMetricsLogger(Callback):
                def on_batch_end(self, batch, logs=None):
                    if 'unet3plus_output_final_activation_loss' in logs \
                        and 'unet3plus_output_final_activation_iou_score' in logs \
                            and 'unet3plus_output_final_activation_precision' in logs \
                            and 'unet3plus_output_final_activation_recall' in logs:
                        print(f"Batch {batch + 1}  Loss = {logs['unet3plus_output_final_activation_loss']:.4f}"
                              f" Precicion = {logs['unet3plus_output_final_activation_precision']:.4f}"
                              f" IoU = {logs['unet3plus_output_final_activation_iou_score']:.4f}"
                              f" Recall = {logs['unet3plus_output_final_activation_recall']:.4f}" )

            self.history = self.model.fit(dataset,
                                          validation_data=val_dataset,
                                          epochs=epochs,
                                          verbose=0,
                                          steps_per_epoch=train_samples // batch_size,
                                          validation_steps=test_samples // batch_size,
                                          callbacks=[early_stop, CustomMetricsLogger()])

            self.model.save(f'models/Unet3Plus_model.h5')

            gc.collect()
        if self.history is not None:
            with open('training_history.json', 'w') as file:
                json.dump(self.history.history, file)
            if show_perf:
                plothistory(self.history.history,)

    def make_predictions(self, val_dataset, batch_size, test_samples):

        model_name = f'Unet3Plus_model.h5'
        model_path = os.path.join('models/', model_name)
        if os.path.exists(model_path):
            model = load_model(model_path, custom_objects={
                'dice_loss_plus_focal_loss': self.dice_loss_plus_focal_loss,
                'iou_score': IOUScore(class_indexes=[0, 1, 2]),
                'precision': Precision(class_indexes=[1]),
                'recall': Recall(class_indexes=[1])
                # other custom objects can be added here
            })
        else:
            raise FileNotFoundError(f"Model file {model_name} not found.")


        # IoU calculation
        #iou_metric = MeanIoU(num_classes=3)
        #for x, y in val_dataset:
        #    y_pred = model.predict(x)
        #    y_pred = np.argmax(y_pred, axis=-1)  # For multi-class
        #    iou_metric.update_state(y, y_pred)

        #iou_score = iou_metric.result().numpy()
        X_test = []
        y_test = []
        num_batches = 1
        for x_batch, y_batch in val_dataset.take(num_batches):
            X_test.append(x_batch.numpy())  # Convert tensor to numpy array
            y_test.append(y_batch.numpy())
        #print(f"Test IoU: {iou_score}")

        sample_image = X_test[0]  # Use one sample image
        prediction = model.predict(sample_image)
        #evaluation_results = model.evaluate(val_dataset, steps=test_samples // batch_size, return_dict=True)
        # y_pred = model.predict(val_dataset, steps=test_samples // batch_size)
        #loss = evaluation_results['loss']
        #print(f'loss {loss}')
        #for metric_name, metric_value in evaluation_results.items():
        #    print(f"{metric_name}: {metric_value}")
        return y_test[0], prediction[0]



