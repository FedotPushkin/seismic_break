import tensorflow as tf
import argparse


def gpu_check():
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


def parser():
    parser = argparse.ArgumentParser(description='First Break detection')
    parser.add_argument('--folder', type=str, help='Path to the input file')
    parser.add_argument('--fit', action='store_true', help='Train the model')
    parser.add_argument('--test', action='store_true', help='Train the model')
    parser.add_argument('--plot-samples', action='store_true', help='Plot image samples')
    parser.add_argument('--load', action='store_true', help='Load signatures from files')
    parser.add_argument('--batch_size',  type=int, default=32, help='Lower if not enough memory')
    parser.add_argument('--epochs', type=int, default=10, help='Lower if not enough memory')
    parser.add_argument('--chunk_size', type=int, default=1100000, help='Lower if not enough memory')
    parser.add_argument("--train_shape", type=int, default=(64, 192), nargs=2,
                        help="Tuple representing dimensions (width, height)")
    args = parser.parse_args()
    return args


def rgb_to_grayscale(rgb):
    return 0.2989 * rgb[:, :, 0] + 0.5870 * rgb[:, :, 1] + 0.1140 * rgb[:, :, 2]
