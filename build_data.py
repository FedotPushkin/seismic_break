import pandas as pd
import itertools
import numpy as np
import tensorflow as tf
from visualisation import show_image_samples
from skimage.transform import resize
import cv2
from tqdm import tqdm
import memory_profiler
import gc

#@memory_profiler.profile


def build_train_data(traces_img, first_break_lines,  im_shape, ds_shape):
    if traces_img is None or first_break_lines is None:
        raise ValueError("Input lists must not be None.")
    if len(traces_img) == 0 or len(first_break_lines) == 0:
        raise ValueError("Input lists must not be empty.")
    if len(traces_img) != len(first_break_lines):
        raise ValueError("Input lists must have equal shapes.")
    else:
        n_samples = len(first_break_lines)
        masks = []



    return traces_img, masks


def build_test_data(images, ev_image, im_size):
    if len(images) == 0 or len(ev_image) == 0:
        raise ValueError("Input lists must not be empty.")
    if len(ev_image) == 1:
        Xt1, Xt2 = [], []
        images = np.reshape(images, (len(images), im_size, im_size, 1))
        ev_image = np.reshape(ev_image, (len(ev_image), im_size, im_size, 1))

        return np.array(Xt1), np.array(Xt2)
    else:
        raise ValueError("Length of eval must be 1.")
