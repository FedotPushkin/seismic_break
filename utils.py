import tensorflow as tf
import argparse
from scipy.signal import butter, filtfilt
from scipy.ndimage import gaussian_filter
import numpy as np
from scipy.signal import wiener, medfilt
import pywt


def gpu_check():
    try:
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f'Num GPUs Available: {len(gpus)}')
        else:
            print("No GPU available. Make sure your TensorFlow installation supports GPU.")
    except RuntimeError as e:
        print(e)


def parser():
    parser = argparse.ArgumentParser(description='First Break detection')
    parser.add_argument('--folder', type=str, help='Path to the input file')
    parser.add_argument('--fit', action='store_true', help='Train the model')
    parser.add_argument('--test', action='store_true', help='Test the model')
    parser.add_argument('--plot-samples', action='store_true', help='Plot image samples')
    parser.add_argument('--load', action='store_true', help='Load signatures from files')
    parser.add_argument('--batch_size',  type=int, default=32, help='Lower if not enough memory')
    parser.add_argument('--chunk_size', type=int, default=1100000, help='Lower if not enough memory')
    parser.add_argument('--epochs', type=int, default=20, help='Lower if not enough memory')
    parser.add_argument('--gaussian', type=float, default=2.0, help='Sigma for Gausian')
    parser.add_argument('--median', type=int, default=5, help='kernel size for median filer')
    parser.add_argument('--weiner', type=int, default=0, help='Window size for Weiner')
    parser.add_argument('--wavelet', type=int, default=0, help='Level for Wavelet')
    parser.add_argument('--bandpass', type=int, default=(0, 60), nargs=2, help='Lower and upper frequency for filter')
    parser.add_argument("--train_shape", type=int, default=(64, 192), nargs=2,
                        help="Tuple representing dimensions (width, height)")
    args = parser.parse_args()
    return args


'''
    Following methods implement various outlier detection,removal and interpolation
'''


def remove_outliers_iqr(data):
    if data is None:
        return
    if len(data) > 0:
        q1 = np.percentile(data, 25)
        q3 = np.percentile(data, 75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        return data[(data >= lower_bound) & (data <= upper_bound)]


def remove_outliers_moving_average(data, window_size=5, threshold=3):
    if data is None or len(data) == 0:
        return data
    if len(data) < window_size:
        window_size = len(data)
    data = data.astype(np.float32)
    # Calculate the moving average and moving standard deviation
    moving_avg = np.convolve(data, np.ones(window_size) / window_size, mode='valid')
    padded_moving_avg = np.pad(moving_avg, (window_size // 2, window_size - 1 - window_size // 2), mode='edge')

    # Calculate the moving standard deviation
    moving_std = np.sqrt(np.convolve((data - padded_moving_avg) ** 2, np.ones(window_size) / window_size, mode='valid'))
    padded_moving_std = np.pad(moving_std, (window_size // 2, window_size - 1 - window_size // 2), mode='edge')

    # Define outliers based on the threshold
    lower_bound = padded_moving_avg - threshold * padded_moving_std
    upper_bound = padded_moving_avg + threshold * padded_moving_std

    # Remove outliers
    non_outliers = np.where((data >= lower_bound) & (data <= upper_bound))[0]
    outliers = np.where((data < lower_bound) | (data > upper_bound))[0]
    if len(non_outliers) == 0:
        return []
    if len(outliers) > 0:
        data[outliers] = np.interp(outliers, non_outliers, data[non_outliers]).astype(int)
    return data


def remove_outliers_z_score(data, threshold=3):
    if data is None or len(data) == 0:
        return data
    data = data.astype(np.float32)
    mean = np.mean(data)
    std = np.std(data)
    if std != 0:
        non_outliers = np.where(np.abs((data - mean) / std) < threshold)[0]
        outliers = np.where(~(np.abs((data - mean) / std) < threshold))[0]
    else:
        return []
    if len(non_outliers) == 0:
        return []
    if len(outliers) > 0:
        try:
            data[outliers] = np.interp(outliers, non_outliers, data[non_outliers]).astype(int)
        except Exception as e:
            print(f"An error occurred: {e}")
    return data


def rgb_to_grayscale(rgb):
    return 0.2989 * rgb[:, :, 0] + 0.5870 * rgb[:, :, 1] + 0.1140 * rgb[:, :, 2]


'''
    Following methods implement various filters for 1d data
'''


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


def compose_filters(image, samp_rate, args):
    samp_rate_hz = 1/(samp_rate * 1e-6)
    for idx, trace in enumerate(image):
        x = trace
        if args.bandpass[0] != 0:
            x = bandpass_filter(x, args.bandpass[0], args.bandpass[1], samp_rate_hz, order=4)
        if args.gaussian > 0:
            x = gaussian_filter(x, sigma=args.gaussian)
        if args.weiner > 0:
            x = wiener(x, mysize=args.weiner)
        if args.wavelet > 0:
            x = wavelet_filter(x, wavelet="db4", level=args.wavelet)
        if args.median > 0:
            x = medfilt(x, kernel_size=args.median)
        image[idx] = x
    return image
