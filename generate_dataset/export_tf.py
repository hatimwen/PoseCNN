from common import ros_to_blender_quat, create_dataset_folder, get_filename_prefix
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
import traceback
import yaml


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


def read_config():
    with open("data/dataset.txt") as f:
        dataset_config = f.readline()
    dataset = os.path.split(dataset_config)[1][:-4]
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
    dataset, boxes, num_boxes, times, start_time, end_time = read_config()

    bag = rosbag.Bag(os.path.join("/home/satco/PycharmProjects/PoseCNN/bag", dataset + ".bag"))
    topics = ["/camera/color/image_raw", "/camera/aligned_depth_to_color/image_raw"]
    tf_t = fill_transformer(bag)
    try:
        os.makedirs(os.path.join("data", dataset))
    except OSError:
        pass

    poses = np.empty((3, 4))
    box_translations = []
    with open(os.path.join("data", dataset, "box_positions.txt"), "w") as f:
        for i in range(num_boxes):
            time = rospy.Time(int(times[i][0]), int(times[i][1]))
            (trans, rot) = tf_t.lookupTransform("vicon", "box" + str(i + 1), time)
            box_translations.append(trans)
            f.write(str(trans + rot) + "\n")
            pose = rot_trans_to_matrix(rot, trans)
            poses = np.dstack((poses, pose))

    cls_indexes = [0] * num_boxes
    cls_indexes = np.float32(cls_indexes)

    # getting added later, just here for completeness sake to show all the fields available in mat_dict
    center = None
    rotation_translation_matrix = None

    poses = np.delete(poses, 0, axis=2)
    vect = np.float32([0, 1])
    vertmap = np.zeros((480, 640, 3))
    intrinsic_matrix = np.float32([[610.55992534, 0, 306.86169342], [0, 610.32086262, 240.94547232], [0, 0, 1]])
    dist = np.float32([[0.10793695], [-0.21546604], [0.00045875], [-0.00670819]])
    mat_dict = {"center": center, "cls_indexes": cls_indexes, "factor_depth": 10000, "intrinsic_matrix":
                intrinsic_matrix, "poses": poses, "rotation_translation_matrix": rotation_translation_matrix,
                "vect": vect, "vertmap": vertmap}

    data_base_path = create_dataset_folder(dataset)

    counter_color = 1
    counter_depth = 1
    bridge = CvBridge()
    f = open(os.path.join("data", dataset, "camera1_positions.txt"), "w")

    print(start_time)
    print(end_time)
    for topic, msg, t in bag.read_messages(topics=topics, start_time=start_time, end_time=end_time):
        if topic == topics[0]:
            if counter_color % 1000 == 0:
                print("Saved 1000 images")
            center = []
            try:
                (trans, rot) = tf_t.lookupTransform("vicon", "camera", msg.header.stamp)
                f.write(str(trans + rot) + "\n")

                rotation_translation_matrix = rot_trans_to_matrix(rot, trans)

                image = bridge.imgmsg_to_cv2(msg, "bgr8")
                prefix = get_filename_prefix(counter_color)
                image_path = os.path.join(data_base_path, prefix + '-color.png')
                meta_mat_path = os.path.join(data_base_path, prefix + '-meta.mat')
                cv2.imwrite(image_path, image)

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
                mat_dict["rotation_translation_matrix"] = rotation_translation_matrix
                sio.savemat(meta_mat_path, mat_dict)
                counter_color += 1
            except tf.ExtrapolationException:
                print("Skipped image")

        if topic == topics[1]:
            # passthrough makes all images very dark
            # image = bridge.imgmsg_to_cv2(msg, "passthrough")
            # looks most like in rviz, but still not the same
            image = bridge.imgmsg_to_cv2(msg, "32FC1")
            image = np.array(image, dtype=np.float64)
            cv2.normalize(image, image, 0, 1, cv2.NORM_MINMAX)
            image = image*255
            # if counter_depth == 2:
            #     image.tofile("test_depth_to_file.txt")
            #     np.savetxt("test_depth.txt", image)
                # np.savetxt("tmp_test_depth_no_passthrough_no_multiply.txt", image)
                # cv2.imwrite("tmp_test_depth.yaml", image)
                # with open("tmp_test_depth.txt", "w") as depth_file:
                #     depth_file.write(image)
            # cv2.imshow("image", image)
            # cv2.waitKey(1)
            prefix = get_filename_prefix(counter_depth)
            image_path = os.path.join(data_base_path, prefix + '-depth.png')
            cv2.imwrite(image_path, image)
            counter_depth += 1

    f.close()


if __name__ == "__main__":
    main()
