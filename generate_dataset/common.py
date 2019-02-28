import os
import yaml
import numpy as np


# blender uses wxyz and ros xyzw
def ros_to_blender_quat(quat):
    return quat[3], quat[0], quat[1], quat[2]


def blender_to_ros_quat(quat):
    return quat[1], quat[2], quat[3], quat[0]


def create_dataset_folder(dataset):
    with open("config.yaml", "r") as config:
        config_dict = yaml.load(config)
    base_path = config_dict["images_path"]
    data_base_path = os.path.join(base_path, dataset)
    try:
        os.makedirs(data_base_path)
    except OSError:
        pass
    return data_base_path


def get_filename_prefix(counter):
    img_number = "0" * (6 - len(str(counter)))
    prefix = img_number + str(counter)
    return prefix


def get_intrinsic_matrix():
    return np.float32([[610.55992534, 0, 306.86169342], [0, 610.32086262, 240.94547232], [0, 0, 1]])


# excpects quat in xyzw form
def invert_quat(quat):
    quat = ros_to_blender_quat(quat)
    norm = float(quat[0]**2 + quat[1]**2 + quat[2]**2 + quat[3]**2)
    quat_inverse = [quat[0]/norm, -quat[1]/norm, -quat[2]/norm, -quat[3]/norm]
    return blender_to_ros_quat(quat_inverse)
