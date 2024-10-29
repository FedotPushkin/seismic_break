from skimage.transform import resize
import numpy as np
from cTab import CTab
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from scipy.interpolate import CubicSpline


def tabToImage(tab, size, anti_aliasing=True):
    if tab is None:
        raise ValueError("tab is None")
    if not isinstance(tab, CTab):
        raise TypeError("Expected CTab instance")
    err_flag, iMinX, iMaxX, iMinY, iMaxY = tab.get_box()
    if not err_flag:
        print('no points in signature')
        return
    if iMinX*iMaxX >= 0:
        width = abs(iMinX - iMaxX)
    else:
        width = abs(iMinX) + abs(iMaxX)
    if iMinY * iMaxY >= 0:
        height = abs(iMinY-iMaxY)
    else:
        height = abs(iMinY) + abs(iMaxY)

    if width < 1:
        print('Invalid tab dimensions: width is zero')
        return
    if height < 1:
        print('Invalid tab dimensions: height is zero')
        return
    img = np.zeros((width + 1, height + 1))
    stp = np.zeros((width + 1, height + 1))
    n_points = tab.get_n_points()
    valid_points = [(tab.get_x(n) - iMinX, tab.get_y(n) - iMinY) for n in range(n_points) if tab.get_y(n) != -1]
    stp[valid_points[0][0], valid_points[0][1]] = 1
    start_points = [(tab.get_x(n) - iMinX, tab.get_y(n) - iMinY) for n in range(1, n_points) if tab.get_y(n-1) == -1]
    end_points = set([(tab.get_x(n) - iMinX, tab.get_y(n) - iMinY) for n in range(n_points-1) if tab.get_y(n + 1) == -1])
    if not valid_points:
        raise ValueError("No valid points in tab")
    points_x, points_y = zip(*valid_points)
    st_x, st_y = zip(*start_points)
    en = zip(*start_points)
    points_x = np.array(points_x)
    points_y = np.array(points_y)
    img[np.array(points_x), np.array(points_y)] = 1
    stp[np.array(st_x), np.array(st_y)] = 1
    start_x, start_y = points_x[0], points_y[0]
    start_points_idx = 0
    lines_counter = 0
    for idx, x in enumerate(points_x):

        y = points_y[idx]
        if start_x == x and start_y == y:
            continue
        if x > width or start_x > width or x < 0 or start_x < 0:
            pass
        if y > height or start_y > height or y < 0 or start_y < 0:
            pass
        try:
            img = cv2.line(img=img, pt1=(start_y, start_x,), pt2=(y, x,), color=1, thickness=2)
            lines_counter += 1
        except Exception as e:
            print(e)

        if (x, y) in end_points:
            start_points_idx += 1
            if start_points_idx < len(st_x):
                start_x, start_y = st_x[start_points_idx], st_y[start_points_idx]
                lines_counter = 0
                continue
        else:
            start_x, start_y = x, y

    img = img.transpose()
    stp = stp.transpose()
    resized_img = resize(img, (size, size), anti_aliasing=anti_aliasing)
    resized_stp = resize(stp, (size, size), anti_aliasing=anti_aliasing)

    img_normalized = (resized_img - resized_img.min()) / (resized_img.max() - resized_img.min())
    stp_normalized = (resized_stp - resized_stp.min()) / (resized_stp.max() - resized_stp.min())

    return img_normalized, stp_normalized

