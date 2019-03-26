from common import get_filename_prefix, get_intrinsic_matrix
from shutil import copyfile
from export_data_from_ros_bag import rot_trans_to_matrix
import json
import os
import numpy as np
import scipy.io as sio
import cv2
import yaml


def main():
    with open("generate_dataset/config.yaml", "r") as config:
        config_dict = yaml.load(config)
    src_folder = config_dict["dope_src"]
    dst_folder = config_dict["dope_dst"]
    intrinsic_matrix = get_intrinsic_matrix()
    for i in range(512, 10000):
        if i % 500 == 0:
            print(i)
        try:
            prefix = get_filename_prefix(i)
            # copy_images
            src_path_prefix = os.path.join(src_folder, prefix)
            dst_path_prefix = os.path.join(dst_folder, prefix)

            try:
                json_f = open(os.path.join(src_folder, prefix + ".json"))
            except IOError:
                continue
            try:
                dope_meta = json.load(json_f)
            except ValueError:
                print(i)
            objects = dope_meta["objects"]
            if not objects:
                continue
            cls_indexes = [1] * len(objects)
            cls_indexes = np.float32(cls_indexes)
            poses = np.empty((3, 4))
            centers = []
            img = cv2.imread(src_path_prefix + ".is.png")
            # set background to zero to be consistent with PoseCNN data
            colors_in_img = list(np.unique(img))
            colors_in_img.remove(0)
            instance_ids = [box["instance_id"] for box in objects]
            background_color = (set(colors_in_img)-set(instance_ids)).pop()
            img[img == background_color] = 0
            for j, box in enumerate(sorted(objects, key=lambda k: k["instance_id"])):
                centers.append(box["projected_cuboid_centroid"])
                # Dope is in cm posecnn in m
                pose = rot_trans_to_matrix(box["quaternion_xyzw"], np.float32(box["location"]) / 100)
                poses = np.dstack((poses, pose))
                if j+1 != box["instance_id"]:
                    img[img == box["instance_id"]] = j+1

            cv2.imwrite(dst_path_prefix + "-object.png", img)
            copyfile(src_path_prefix + ".png", dst_path_prefix + "-color.png")
            copyfile(src_path_prefix + ".cs.png", dst_path_prefix + "-label.png")
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
            import traceback
            print(traceback.format_exc())
            print(i)
            json_f.close()
            exit()


if __name__ == '__main__':
    main()