import argparse
import tensorflow as tf
import numpy as np
import os
from nn import Siamese_NN
from build_data import create_fit_data, create_test_data
from loading import load_form_and_truth, load_signatures
from visualisation import show_image_samples, show_performance_metrics, make_predictions

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Siamese Neural Network for Signature Verification')
    parser.add_argument('--f', action='store_true', help='Train the model')
    parser.add_argument('--t', action='store_true', help='Train the model')
    parser.add_argument('--plot-samples', action='store_true', help='Plot image samples')
    parser.add_argument('--load', action='store_true', help='Load signatures from files')
    parser.add_argument('--genuine_folder', type=str, help='Path to the input file')
    parser.add_argument('--test_path', type=str, help='Path to the input file')
    parser.add_argument('--eval_folder', type=str, help='Path to the input file')
    args = parser.parse_args()
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f'Num GPUs Available: {len(gpus)}')
        except RuntimeError as e:
            print(e)
    else:
        print("No GPU available. Make sure your TensorFlow installation supports GPU.")

    fit = args.f
    test = args.t
    genuine_folder = args.genuine_folder
    eval_folder = args.eval_folder
    # test_path = args.test_path
    plot_samples = args.plot_samples
    load = args.load
    im_size = 128
    test_size = 0.1

    if test:
        genuine_paths = [f for f in os.listdir(genuine_folder)]
        eval_paths = [f for f in os.listdir(eval_folder)]
        y_eval = []
        y_true = []
        for idx, genuine_path in enumerate(genuine_paths):
            try:
                eval_path = eval_paths[idx]
                genuine_path = os.path.join(genuine_folder, genuine_path)
                eval_path = os.path.join(eval_folder, eval_path)
                sample_ids, images, stps = load_signatures(genuine_path, im_size, True, test)
                eval_id, ev_image, ev_stps = load_signatures(eval_path, im_size, True, test)
                if plot_samples:
                    show_image_samples(images+ev_image, sample_ids+eval_id)
            except Exception as e:
                print(f"Error loading signatures: {e}")
                raise
            Xt1, Xt2 = create_test_data(images, stps, ev_image, ev_stps, im_size)
            y_pred_mean = np.mean(make_predictions(Xt1, Xt2))
            _, truth = load_form_and_truth(eval_id)
            y_eval.append(y_pred_mean)
            y_true.append(truth[0])
            print(f'signature is {y_pred_mean*100:.2f} % genuine')
        show_performance_metrics(np.array(y_true), np.array(y_eval))
    else:
        folder_path = 'images'
        try:
            sample_ids, images, stps = load_signatures(folder_path, im_size, load, test)
        except Exception as e:
            print(f"Error loading signatures: {e}")
            raise

        if plot_samples:
            show_image_samples(images[:6], sample_ids[:6])

        forms, truth = load_form_and_truth(sample_ids)

        Xt1, Xt2, y = create_fit_data(sample_ids, images, stps, forms, truth, im_size)

        model = Siamese_NN(input_shape=(im_size, im_size, 2))

        if fit:
            model.fit_to_data(Xt1, Xt2, y, batch_size=16, epochs=20, show_perf=True)







