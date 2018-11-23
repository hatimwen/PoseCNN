import os
import yaml


# blender uses wxyz and ros xyzw
def ros_to_blender_quat(qaut):
    return qaut[-1], qaut[0], qaut[1], qaut[2]


def create_dataset_folder(dataset):
    with open("config.yaml", "r") as config:
        config_dict = yaml.load(config)
    base_path = config_dict["datasets_base_path"]
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
