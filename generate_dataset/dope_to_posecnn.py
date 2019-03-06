from common import get_filename_prefix, get_intrinsic_matrix
from shutil import copyfile
from export_data_from_ros_bag import rot_trans_to_matrix
import json
import os
import numpy as np
import scipy.io as sio


def main():
    src_folder = "/home/satco/catkin_ws/src/Deep_Object_Pose/data/Static"
    dst_folder = "/home/satco/remote_root/mnt/drive_c/datasets/kaju/PoseCNN/data/LOV/data/dope"
    intrinsic_matrix = get_intrinsic_matrix()
    for i in range(5000):
        if i % 1000 == 0:
            print(i)
        try:
            prefix = get_filename_prefix(i)
            # copy_images
            src_path_prefix = os.path.join(src_folder, prefix)
            dst_path_prefix = os.path.join(dst_folder, prefix)

            json_f = open(os.path.join(src_folder, prefix + ".json"))
            try:
                dope_meta = json.load(json_f)
            except ValueError:
                print(i)
            objects = dope_meta["objects"]
            cls_indexes = [1] * len(objects)
            cls_indexes = np.float32(cls_indexes)
            poses = np.empty((3, 4))
            centers = []
            for box in objects:
                centers.append(box["projected_cuboid_centroid"])
                # Dope is in cm posecnn in m
                pose = rot_trans_to_matrix(box["quaternion_xyzw"], np.float32(box["location"])/100)
                poses = np.dstack((poses, pose))

            try:
                centers = [centers[1], centers[2], centers[0]]
            except IndexError:
                continue

            copyfile(src_path_prefix + ".png", dst_path_prefix + "-color.png")
            copyfile(src_path_prefix + ".cs.png", dst_path_prefix + "-label.png")
            copyfile(src_path_prefix + ".is.png", dst_path_prefix + "-object.png")
            copyfile(src_path_prefix + ".depth.png", dst_path_prefix + "-depth.png")

            centers = np.float32(centers)
            poses = np.delete(poses, 0, axis=2)
            rotation_translation_matrix = np.zeros((3,4))
            vertmap = np.zeros((480, 640, 3))

            mat_dict = {"center": centers, "cls_indexes": cls_indexes, "factor_depth": 10000, "intrinsic_matrix": intrinsic_matrix, "poses": poses,
                        "rotation_translation_matrix": rotation_translation_matrix, "vertmap": vertmap}
            meta_mat_path = os.path.join(dst_path_prefix + "-meta.mat")
            sio.savemat(meta_mat_path, mat_dict)
            json_f.close()
        except Exception as e:
            print(e)
            print(i)
            json_f.close()
            exit()


if __name__ == '__main__':
    main()