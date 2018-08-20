import numpy as np


def object_to_camera_tf(x_camera, x_object):
    return x_object - x_camera


def three_to_2d(K, x_camera):
    return K*x_camera
