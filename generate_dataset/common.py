import os


# blender uses wxyz and ros xyzw
def ros_to_blender_quat(qaut):
    return qaut[-1], qaut[0], qaut[1], qaut[2]


def create_dataset_folder(dataset):
    data_base_path = os.path.join("/media/satco/My Passport/Uni/Master thesis/data", dataset)
    try:
        os.makedirs(data_base_path)
    except OSError:
        pass
    return data_base_path


def get_filename_prefix(counter):
    img_number = "0" * (6 - len(str(counter)))
    prefix = img_number + str(counter)
    return prefix
