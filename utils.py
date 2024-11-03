import tensorflow as tf
import argparse
from scipy.signal import butter, filtfilt
from scipy.ndimage import gaussian_filter
import numpy as np
from scipy.signal import wiener, medfilt
import pywt


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
    parser.add_argument('--chunk_size', type=int, default=1100000, help='Lower if not enough memory')
    parser.add_argument('--epochs', type=int, default=6, help='Lower if not enough memory')
    parser.add_argument('--gaussian', type=float, default=0.0, help='Sigma for Gausian')
    parser.add_argument('--median', type=int, default=0, help='kernel size for median filer')
    parser.add_argument('--weiner', type=int, default=0, help='Window size for Weiner')
    parser.add_argument('--wavelet', type=int, default=0, help='Level for Wavelet')
    parser.add_argument('--bandpass', type=int, default=(10, 60), nargs=2, help='Lower and upper frequency for filter')
    parser.add_argument("--train_shape", type=int, default=(64, 192), nargs=2,
                        help="Tuple representing dimensions (width, height)")
    args = parser.parse_args()
    return args


def rgb_to_grayscale(rgb):
    return 0.2989 * rgb[:, :, 0] + 0.5870 * rgb[:, :, 1] + 0.1140 * rgb[:, :, 2]


def bandpass_filter(data, lowcut, highcut, fs, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(N=order, Wn=[low, high], btype="band", fs=fs)
    return filtfilt(b, a, data)


def wavelet_filter(data, wavelet="db4", level=2):
    coeffs = pywt.wavedec(data, wavelet, level=level)
    coeffs[1:] = [np.zeros_like(i) for i in coeffs[1:]]
    return pywt.waverec(coeffs, wavelet)


def compose_filters(data, samp_rate, args):
    x = data
    if args.bandpass[0] != 0:
        x = bandpass_filter(data, args.bandpass[0], args.bandpass[1], samp_rate, order=4)
    if args.gaussian > 0:
        x = gaussian_filter(x, sigma=args.gaussian)
    if args.weiner > 0:
        x = wiener(x, mysize=args.weiner)
    if args.wavelet > 0:
        x = wavelet_filter(x, wavelet="db4", level=args.wavelet)
    if args.median > 0:
        x = medfilt(x, kernel_size=args.median)

    return x
