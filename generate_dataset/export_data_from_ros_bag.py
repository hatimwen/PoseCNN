from common import ros_to_blender_quat, get_filename_prefix
import cv2
from cv_bridge import CvBridge
import geometry_msgs
import os
from pyquaternion import Quaternion
import numpy as np
import rosbag
import rospy
import scipy.io as sio
import tf
import yaml


class Dataset(object):
    bags_path = ""
    data_output_path = ""

    def __init__(self, name, boxes_sizes, num_boxes, times, start_time, end_time):
        self.name = name
        self.boxes_sizes = boxes_sizes
        self.num_boxes = num_boxes
        self.times = times
        self.start_time = start_time
        self.end_time = end_time


def bag_type_to_geometry_msgs(msg_tf):
    casted_msg = geometry_msgs.msg.TransformStamped()
    casted_msg.header = msg_tf.header
    casted_msg.child_frame_id = msg_tf.child_frame_id
    casted_msg.transform.translation.x = msg_tf.transform.translation.x
    casted_msg.transform.translation.y = msg_tf.transform.translation.y
    casted_msg.transform.translation.z = msg_tf.transform.translation.z
    casted_msg.transform.rotation.x = msg_tf.transform.rotation.x
    casted_msg.transform.rotation.y = msg_tf.transform.rotation.y
    casted_msg.transform.rotation.z = msg_tf.transform.rotation.z
    casted_msg.transform.rotation.w = msg_tf.transform.rotation.w
    return casted_msg


def fill_transformer(bag):
    print("Loading tfs into transformer...")
    tf_t = tf.Transformer(True, rospy.Duration(3600))
    for topic, msg, t in bag.read_messages(topics=["/tf"]):
        for msg_tf in msg.transforms:
            casted_msg = bag_type_to_geometry_msgs(msg_tf)
            tf_t.setTransform(casted_msg)
    print("Finished")
    return tf_t


def get_datasets():
    datasets = []
    with open("config.yaml", "r") as config:
        config_dict = yaml.load(config)
    Dataset.bags_path = config_dict["bags_path"]
    Dataset.data_output_path = config_dict["images_path"]
    with open("data/dataset.txt") as datasets_f:
        for line in datasets_f:
            dataset_path = line.rstrip()
            if not line.startswith(";"):
                dataset = os.path.split(dataset_path)[1][:-5]
                stream = open(dataset_path, "r")
                yaml_data = yaml.load_all(stream)
                data_dict = list(yaml_data)[0]
                boxes_sizes = data_dict["boxes"]
                num_boxes = len(boxes_sizes)
                with open("data/" + dataset + "/times.txt") as times_f:
                    times = []
                    # plus 2 for start and stop time
                    for i in range(num_boxes + 2):
                        times.append(times_f.readline().split("."))
                start_time = rospy.Time(int(times[-2][0]), int(times[-2][1]))
                end_time = rospy.Time(int(times[-1][0]), int(times[-1][1]))
                datasets.append(Dataset(dataset, boxes_sizes, num_boxes, times, start_time, end_time))
    return datasets


def read_config():
    with open("data/dataset.txt") as f:
        dataset_config = f.readline().rstrip()
    dataset = os.path.split(dataset_config)[1][:-5]
    stream = open(dataset_config, "r")
    yaml_data = yaml.load_all(stream)
    data_dict = list(yaml_data)[0]
    boxes = data_dict["boxes"]
    num_boxes = len(boxes)
    with open("data/" + dataset + "/times.txt") as f:
        times = []
        # plus 2 for start and stop time
        for i in range(num_boxes + 2):
            times.append(f.readline().split("."))
    start_time = rospy.Time(int(times[-2][0]), int(times[-2][1]))
    end_time = rospy.Time(int(times[-1][0]), int(times[-1][1]))
    return dataset, boxes, num_boxes, times, start_time, end_time


def rot_trans_to_matrix(rot, trans):
    quat = Quaternion(ros_to_blender_quat(rot))
    rot = quat.rotation_matrix
    trans = np.float32([[trans[0]], [trans[1]], [trans[2]]])
    pose = np.hstack((rot, trans))
    return pose


def main():

    export_tf = True
    export_images = True
    export_meta = True
    export_depth = True
    s_tf = "tf data" if export_tf else ""
    s_images = "color images" if export_images else ""
    s_meta = "meta data" if export_meta else ""
    s_depth = "depth images" if export_depth else ""

    datasets = get_datasets()

    print("Writing {} {} {} {} into {}".format(s_tf, s_images, s_meta, s_depth, Dataset.data_output_path))
    for dataset in datasets:
        print("Preparing " + dataset.name)
        num_boxes = dataset.num_boxes
        times = dataset.times
        start_time = dataset.start_time
        end_time = dataset.end_time
        bag = rosbag.Bag(os.path.join(Dataset.bags_path, dataset.name + ".bag"))
        topics = ["/camera/color/image_raw", "/camera/aligned_depth_to_color/image_raw"]
        tf_t = fill_transformer(bag)
        try:
            os.makedirs(os.path.join("data", dataset.name))
        except OSError:
            pass

        if export_meta:
            box_translations = []
            with open(os.path.join("data", dataset.name, "box_positions.txt"), "w") as f:
                for i in range(num_boxes):
                    time = rospy.Time(int(times[i][0]), int(times[i][1]))
                    (trans, rot) = tf_t.lookupTransform("vicon", "box" + str(i + 1), time)
                    box_translations.append(trans)
                    f.write(str(trans + rot) + "\n")

            cls_indexes = [1] * num_boxes
            cls_indexes = np.float32(cls_indexes)

            # getting added later, just here for completeness sake to show all the fields available in mat_dict
            center = None
            rotation_translation_matrix = None
            poses = None
            vertmap = np.zeros((480, 640, 3))
            intrinsic_matrix = np.float32([[610.55992534, 0, 306.86169342], [0, 610.32086262, 240.94547232], [0, 0, 1]])
            dist = np.float32([[0.10793695], [-0.21546604], [0.00045875], [-0.00670819]])
            mat_dict = {"center": center, "cls_indexes": cls_indexes, "factor_depth": 10000, "intrinsic_matrix":
                        intrinsic_matrix, "poses": poses, "rotation_translation_matrix": rotation_translation_matrix, "vertmap": vertmap}

        try:
            os.makedirs(os.path.join(Dataset.data_output_path, dataset.name))
        except OSError:
            pass

        counter_color = 1
        counter_depth = 1
        if export_images or export_depth:
            bridge = CvBridge()
        if export_tf:
            f = open(os.path.join("data", dataset.name, "camera1_positions.txt"), "w")

        print(start_time)
        print(end_time)
        for topic, msg, t in bag.read_messages(topics=topics, start_time=start_time, end_time=end_time):
            if counter_color % 1000 == 0:
                print("Saved 1000 images")
            if topic == topics[0]:
                center = []
                try:
                    if export_tf or export_meta:
                        (trans, rot) = tf_t.lookupTransform("vicon", "camera", msg.header.stamp)
                        if export_tf:
                            f.write(str(trans + rot) + "\n")

                    prefix = get_filename_prefix(counter_color)
                    if export_images:
                        image_path = os.path.join(Dataset.data_output_path, dataset.name, prefix + '-color.png')
                        image = bridge.imgmsg_to_cv2(msg, "bgr8")
                        cv2.imwrite(image_path, image)

                    if export_meta:
                        (trans_cam, rot_cam) = tf_t.lookupTransform("camera", "vicon", msg.header.stamp)
                        camera_quat = Quaternion(ros_to_blender_quat(rot_cam))
                        camera_rot = camera_quat.rotation_matrix
                        camera_rodrigues, jacobian = cv2.Rodrigues(camera_rot)
                        for i, trans in enumerate(box_translations):
                            boxes_project_points, jacobian = cv2.projectPoints(np.float32([trans]), camera_rodrigues,
                                                                               np.float32(trans_cam), intrinsic_matrix, dist)
                            boxes_project_points = boxes_project_points[0]
                            boxes_project_points = boxes_project_points[0]
                            center.append(boxes_project_points)

                        center = np.float32(center)
                        mat_dict["center"] = center
                        rotation_translation_matrix = rot_trans_to_matrix(rot, trans)
                        mat_dict["rotation_translation_matrix"] = rotation_translation_matrix
                        poses = np.empty((3, 4))
                        for i in range(num_boxes):
                            (trans, rot) = tf_t.lookupTransform("camera", "box" + str(i + 1) + "_static", msg.header.stamp)
                            pose = rot_trans_to_matrix(rot, trans)
                            poses = np.dstack((poses, pose))
                        poses = np.delete(poses, 0, axis=2)
                        mat_dict["poses"] = poses
                        meta_mat_path = os.path.join(Dataset.data_output_path, dataset.name, prefix + '-meta.mat')
                        sio.savemat(meta_mat_path, mat_dict)
                    counter_color += 1
                except tf.ExtrapolationException:
                    print("Skipped image")

            if topic == topics[1]:
                # passthrough makes all images very dark
                # image = bridge.imgmsg_to_cv2(msg, "passthrough")
                # looks most like in rviz, but still not the same
                if export_depth:
                    image = bridge.imgmsg_to_cv2(msg, "32FC1")
                    image = np.array(image, dtype=np.float64)
                    cv2.normalize(image, image, 0, 1, cv2.NORM_MINMAX)
                    image = image*255
                    prefix = get_filename_prefix(counter_depth)
                    image_path = os.path.join(Dataset.data_output_path, dataset.name, prefix + '-depth.png')
                    cv2.imwrite(image_path, image)
                counter_depth += 1

        if export_tf:
            f.close()


if __name__ == "__main__":
    main()
