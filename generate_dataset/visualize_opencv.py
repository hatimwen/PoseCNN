from common import ros_to_blender_quat, get_filename_prefix
import cv2
from cv_bridge import CvBridge
from export_tf import fill_transformer, get_datasets, Dataset
import numpy as np
from pyquaternion import Quaternion
import rosbag
import rospy
import tf
import os


def get_corner(tf_t, times, num_boxes):
    corner = []
    for i in range(num_boxes):
        (trans, rot) = tf_t.lookupTransform("vicon", "box" + str(i + 1), rospy.Time(int(times[i][0]), int(times[i][1])))
        corner.append(list(trans))
    corners = np.float32(corner)
    return corners


def calculate_point(name, K, R, t, corner):
    print(name + "\n\n")
    print("K: " + str(K))
    R = np.hstack((R, t))
    print("R: " + str(R))
    print("T: " + str(t))
    print("Corner: " + str(corner))
    result = np.dot(np.dot(K, R), corner)
    print("By hand: " + str(result))


def calculate_point_camera(name, K, corner):
    print(name + "\n\n")
    print("K: " + str(K))
    print("Corner: " + str(corner))
    result = np.dot(K, corner)
    print("By hand camera frame: " + str(result))


def main():
    datasets = get_datasets()
    # dataset, boxes, num_boxes, times, start_time, end_time = read_config()
    for dataset in datasets:
        times = dataset.times
        num_boxes = dataset.num_boxes
        start_time = dataset.start_time
        end_time = dataset.end_time
        bag = rosbag.Bag(os.path.join(Dataset.bags_path, dataset.name + ".bag"))
        topics = ["/camera/color/image_raw"]
        tf_t = fill_transformer(bag)
        corner_input = get_corner(tf_t, times, num_boxes)
        # print(corner_input)
        camera_matrix = np.float32([[610.55992534, 0, 306.86169342], [0, 610.32086262, 240.94547232], [0, 0, 1]])
        dist = np.float32([[0.10793695], [-0.21546604], [0.00045875], [-0.00670819]])
        # dist = np.float32([[0], [0], [0], [0]])
        bridge = CvBridge()
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 0.5
        fontColor = (255, 255, 255)
        lineType = 2

        print(start_time)
        print(end_time)
        counter = 1
        for topic, msg, t in bag.read_messages(topics=topics, start_time=start_time, end_time=end_time):
            # print(msg.header.stamp)
            if topic == "/camera/color/image_raw":
                try:
                    (trans, rot) = tf_t.lookupTransform("camera", "vicon", msg.header.stamp)
                except tf.ExtrapolationException:
                    pass
                trans = np.float32([[trans[0]], [trans[1]], [trans[2]]])
                camera_quat = Quaternion(ros_to_blender_quat(rot))
                camera_rodrigues, jacobian = cv2.Rodrigues(camera_quat.rotation_matrix)
                image = bridge.imgmsg_to_cv2(msg, "bgr8")
                corners, jacobian = cv2.projectPoints(corner_input, camera_rodrigues, trans, camera_matrix, dist)
                corner_counter = 1
                for corner in corners:
                    corner = tuple(corner[0])
                    if corner[0] > 0 and corner[1] > 0:
                        cv2.circle(image, corner, 3, (255, 0, 0), -1)
                        cv2.putText(image, str(corner_counter), corner, font, fontScale, fontColor, lineType)
                        corner_counter += 1
                prefix = get_filename_prefix(counter)
                mask_name = os.path.join(Dataset.data_output_path, dataset.name, prefix + "-label.png")
                mask = cv2.imread(mask_name)
                alpha = 0.5
                image_with_mask = cv2.addWeighted(mask, alpha, image, 1 - alpha, 0)
                w = 1280
                h = 960
                image_with_mask = cv2.resize(image_with_mask, (w, h))
                cv2.resizeWindow("Image", w, h)
                cv2.imshow("Image", image_with_mask)
                cv2.waitKey(100)
                counter += 1


if __name__ == "__main__":
    main()
