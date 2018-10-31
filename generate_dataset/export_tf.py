import cv2
from cv_bridge import CvBridge
import geometry_msgs
import os
import rosbag
import rospy
import tf
import yaml
import traceback


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
    dataset = os.path.split(dataset_config)[1].split(".")[0]
    stream = open(dataset_config, "r")
    yaml_data = yaml.load_all(stream)
    data_dict = list(yaml_data)[0]
    boxes = data_dict["boxes"]
    return dataset, boxes


def main():
    dataset, boxes = read_config()
    num_boxes = len(boxes)
    with open("data/" + dataset + "/times.txt") as f:
        times = []
        for i in range(num_boxes):
            times.append(f.readline().split("."))
    bag = rosbag.Bag(os.path.join("/home/satco/PycharmProjects/PoseCNN/bag", dataset + ".bag"))
    topics = ["/camera/color/image_raw", "/camera/aligned_depth_to_color/image_raw"]
    tf_t = fill_transformer(bag)
    try:
        os.makedirs(os.path.join("data", dataset))
    except OSError:
        pass
    with open(os.path.join("data", dataset, "box_positions.txt"), "w") as f:
        for i in range(num_boxes):
            (trans, rot) = tf_t.lookupTransform("vicon", "box" + str(i + 1), rospy.Time(int(times[i][0]), int(times[i][1])))
            f.write(str(trans + rot) + "\n")

    data_base_path = os.path.join("/media/satco/My Passport/Uni/Master thesis/data", dataset)
    try:
        os.makedirs(data_base_path)
    except OSError:
        pass

    counter_color = 1
    counter_depth = 1
    bridge = CvBridge()
    f = open(os.path.join("data", dataset, "camera1_positions.txt"), "w")
    for topic, msg, t in bag.read_messages(topics=topics, start_time=rospy.Time(int(times[-1][0]), int(times[-1][1]))):
        if topic == topics[0]:
            try:
                (trans, rot) = tf_t.lookupTransform("vicon", "camera", msg.header.stamp)
                f.write(str(trans + rot) + "\n")
            except tf.ExtrapolationException:
                print(traceback.format_exc())
                print("Skipped image")
        #     image = bridge.imgmsg_to_cv2(msg, "bgr8")
        #     img_number = "0" * (6 - len(str(counter_color)))
        #     image_path = os.path.join(data_base_path, img_number + str(counter_color) + '-color.png')
        #     cv2.imwrite(image_path, image)
        #     counter_color += 1
        # if topic == topics[1]:
        #     image = bridge.imgmsg_to_cv2(msg, "bgr8")
        #     img_number = "0" * (6 - len(str(counter_depth)))
        #     image_path = os.path.join(data_base_path, img_number + str(counter_color) + '-depth.png')
        #     cv2.imwrite(image_path, image)
        #     counter_depth += 1

    f.close()


if __name__ == "__main__":
    main()
