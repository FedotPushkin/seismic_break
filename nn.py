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


class Siamese_NN:

    def __init__(self, input_shape=(256, 256, 1)):

        self.x1_train, self.x1_test, self.x2_train, self.x2_test, self.y_train, self.y_test = \
            None, None, None, None, None, None
        self.X1 = None
        self.X2 = None
        self.y = None
        self.im_size = input_shape[0]
        self.history = None

        #base_network = self.create_big_network(input_shape)
        base_network = self.setup_resnet(input_shape)
        self.model = self.setup_model(input_shape, base_network)

    def setup_resnet(self, input_shape):

        base_network = MobileNetV2(weights='imagenet', include_top=False, input_shape=(self.im_size, self.im_size, 3))
        new_input = Input(shape=input_shape)

        #base_network = keras_resnet.models.ResNet18(inputs=input3, classes=2)
        x = Conv2D(3, (1, 1))(new_input)
        for layer in base_network.layers:
            layer.trainable = False
        for layer in base_network.layers[-3:]:
            layer.trainable = True
        x = base_network(x)
        x = GlobalAveragePooling2D()(x)
        predictions = Dense(1024, activation='relu')(x)  # Add a dense layer if needed
        #predictions = Dense(2, activation='softmax')(x)
        model = Model(inputs=new_input, outputs=predictions)
        return model

    @staticmethod
    def setup_model(input_shape, base_network):

        # Входные данные: два изображения подписей
        input_a = layers.Input(shape=input_shape)
        input_b = layers.Input(shape=input_shape)

        # Извлекаем признаки для обеих подписей
        processed_a = base_network(input_a)
        processed_b = base_network(input_b)

        # Вычисляем расстояние между двумя векторами признаков
        #distance = layers.Lambda(lambda tensors: tf.math.abs(tensors[0] - tensors[1]))([processed_a, processed_b])
        distance = layers.Lambda(lambda tensors: tf.sqrt(
            tf.maximum(tf.reduce_sum(tf.square(tensors[0] - tensors[1]), axis=1, keepdims=True),
                       tf.keras.backend.epsilon())
        ))([processed_a, processed_b])

        # Добавляем полносвязный слой для классификации (подлинная или поддельная подпись)
        outputs = layers.Dense(1, activation='sigmoid')(distance)

        # Модель сиамской нейросети
        model = models.Model([input_a, input_b], outputs)


        # Компиляция модели
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        return model





    def fit_to_data(self, Xt1, Xt2, y,batch_size, epochs, show_perf):
        k = 5
        kf = KFold(n_splits=k, shuffle=True, random_state=42)
        self.X1 = np.asarray(Xt1, dtype=np.float16)
        self.X2 = np.asarray(Xt2, dtype=np.float16)
        self.y = y
        with tf.device('/GPU:0'):
            gc.collect()

            if not os.path.exists('models'):
                os.makedirs('models')
            # Batch and prefetch

            fold = 0
            for train_index, val_index in kf.split(self.X1):

                X1_train, X1_val = self.X1[train_index], self.X1[val_index]
                X2_train, X2_val = self.X2[train_index], self.X2[val_index]
                y_train, y_val = self.y[train_index], self.y[val_index]

                train_distr = (np.count_nonzero(y_train == 1) / len(y_train)) * 100
                test_distr = (np.count_nonzero(y_val == 1) / len(y_val)) * 100
                print(f'positive examples % in train set {train_distr}')
                print(f'positive examples % in test set {test_distr}')
                tf.keras.backend.set_floatx('float16')
                class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train),
                                                     y=y_train)
                class_weight_dict = dict(enumerate(class_weights))

                dataset, val_set = self.get_datasets(X1_train, X2_train, y_train, X1_val, X2_val, y_val, batch_size)
                gc.collect()
                early_stop = keras.callbacks.EarlyStopping(monitor='val_loss',
                                                           patience=5,
                                                           verbose=1,
                                                           restore_best_weights=True
                                                           )
                self.history = self.model.fit(dataset,
                                              validation_data=val_set,
                                              epochs=epochs,
                                              steps_per_epoch=len(y_train) // batch_size,
                                              validation_steps=len(y_val) // batch_size,
                                              callbacks=[early_stop],
                                              class_weight=class_weight_dict,)

                self.model.save(f'models/my_model_f16_k5_l1_fold{fold}.h5')
                fold += 1
                gc.collect()
        if self.history is not None:
            if show_perf:
                plothistory(self.history.history)
                y_pred = make_predictions(self.x1_test, self.x2_test)
                show_performance_metrics(self.y_test, y_pred)

    def custom_generator(self,X1, X2, y, batch_size, train_generator):
        num_samples = X1.shape[0]
        while True:
            indices = np.random.permutation(num_samples)  # Shuffle indices
            for start in range(0, num_samples, batch_size):
                end = min(start + batch_size, num_samples)
                batch_indices = indices[start:end]

                # Apply augmentation to both input images
                X1_augmented = train_generator.flow(X1[batch_indices], batch_size=batch_size, shuffle=False)
                X2_augmented = train_generator.flow(X2[batch_indices], batch_size=batch_size, shuffle=False)

                # Get the augmented images
                X1_batch = next(X1_augmented)
                X2_batch = next(X2_augmented)

                # Yield the two augmented input images and their labels
                yield (X1_batch, X2_batch), y[batch_indices]

    def get_datasets(self,X1_t, X2_t, y_t, X1_v, X2_v, y_v, batch_size):

        im_gen = ImageDataGenerator(
            rotation_range=10,
            width_shift_range=0.05,
            height_shift_range=0.05,
            shear_range=0.01,
            zoom_range=[0.95, 1.05],

            fill_mode='nearest'
        )
        train_gen = self.custom_generator(X1_t, X2_t, y_t, batch_size, im_gen)
        val_gen = self.custom_generator(X1_v, X2_v, y_v, batch_size, ImageDataGenerator())
        signature = ((tf.TensorSpec(shape=(None, self.im_size, self.im_size, 2), dtype=tf.float16),
                      tf.TensorSpec(shape=(None, self.im_size, self.im_size, 2), dtype=tf.float16)),
                     tf.TensorSpec(shape=(None,), dtype=tf.float16))

        train_dataset = tf.data.Dataset.from_generator(lambda: train_gen, output_signature=signature). \
            prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

        validation_set = tf.data.Dataset.from_generator(lambda: val_gen, output_signature=signature). \
            prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        return train_dataset, validation_set
