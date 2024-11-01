import argparse
import tensorflow as tf
from nn import Unet_NN
import cProfile
from build_data import build_train_data, build_test_data
from loading import load_db

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='First Break detection')
    parser.add_argument('--f', action='store_true', help='Train the model')
    parser.add_argument('--t', action='store_true', help='Train the model')
    parser.add_argument('--plot-samples', action='store_true', help='Plot image samples')
    parser.add_argument('--load', action='store_true', help='Load signatures from files')
    parser.add_argument('--folder', type=str, help='Path to the input file')
    parser.add_argument('--eval_folder', type=str, help='Path to the eval input file')

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
    folder = args.folder
    eval_folder = args.eval_folder
    plot_samples = args.plot_samples
    load = args.load
    train_shape = (64, 192)
    test_size = 0.2

    if fit:

       # profiler = cProfile.Profile()

        try:
            traces_img, masks = load_db(folder, train_shape, load, test)

        except Exception as e:
            print(f"Error loading images: {e}")
            raise

        #if plot_samples:
        #    show_image_samples(images[:6], sample_ids[:6])
        #profiler.enable()

        #X, y = build_train_data(traces_img, masks,
        #                        im_shape=train_shape,
        #                        ds_shape=ds_shape)
        #profiler.disable()
        #profiler.print_stats(sort='time')
        model = Unet_NN(input_shape=(train_shape[0], train_shape[1], 1))

        if fit:

            model.fit_to_data(traces_img, masks, batch_size=16, epochs=20, show_perf=True)







