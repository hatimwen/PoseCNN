from cv_bridge import CvBridge
import cv2
import os
import rosbag
import rospy
import yaml
import subprocess


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


def main():
    subprocess.call(["rviz"])
    subprocess.call(["roslaunch "])
    datasets = get_datasets()
    bridge = CvBridge()
    for dataset in datasets:
        print("Preparing " + dataset.name)
        num_boxes = dataset.num_boxes
        times = dataset.times
        start_time = dataset.start_time
        end_time = dataset.end_time
        bag_path = os.path.join(Dataset.bags_path, dataset.name + ".bag")
        print("Opening " + bag_path)
        bag = rosbag.Bag(bag_path)
        topics = ["/camera/color/image_raw", "/camera/aligned_depth_to_color/image_raw", "/tf"]
        for topic, msg, t in bag.read_messages(topics=topics):
            if topic == "/tf":
                print(msg)
            else:
                image = bridge.imgmsg_to_cv2(msg, "bgr8")
                cv2.imshow("Image", image)
                cv2.waitKey(10)


def change_message_to_static_transform(msg):
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


def main2():
    bag = rosbag.Bag("../bag/dataset1.6.bag")
    bridge = CvBridge()
    topics = ["/camera/color/image_raw", "/camera/aligned_depth_to_color/image_raw", "/tf"]
    for topic, msg, t in bag.read_messages(topics=topics):
        if topic == "/tf":
            print(msg)
            new_msg = change_message_to_static_transform(msg.transform)
            bag.write("/tf", , t)
            bag.write
        else:
            image = bridge.imgmsg_to_cv2(msg, "bgr8")
            cv2.imshow("Image", image)
            cv2.waitKey(10)



if __name__ == '__main__':
    main2()
