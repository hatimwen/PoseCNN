import rosbag
import rospy
import tf
import geometry_msgs


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


def main():
    # bag = rosbag.Bag("/home/satco/PycharmProjects/PoseCNN/bag/dataset_one_box.bag")
    bag = rosbag.Bag("/home/satco/PycharmProjects/PoseCNN/bag/test.bag")
    # topics = ["/camera1/color/image_raw", "/camera2/color/image_raw"]
    topics = ["/camera/color/image_raw"]
    tf_t = fill_transformer(bag)
    (trans, rot) = tf_t.lookupTransform("vicon", "box1", rospy.Time(1536832337, 186568))

    with open("data/box_positions.txt", "w") as f:
        f.write(str(trans + rot) + "\n")
    f = open("data/camera1_positions.txt", "w")
    # f2 = open("data/timestamps2.txt", "w")
    timestamps = []
    timestamps2 = []
    for topic, msg, t in bag.read_messages(topics=topics, start_time=rospy.Time(1536832352, 412808)):
        print(msg.header.stamp)
        # if topic == "/camera1/color/image_raw":
        if topic == "/camera/color/image_raw":
            timestamps.append(msg.header.stamp)
            # print(msg.header.stamp)
        if topic == "/camera2/color/image_raw":
            timestamps2.append(msg.header.stamp)

    for timestamp in timestamps:
        (trans, rot) = tf_t.lookupTransform("vicon", "camera", timestamp)
        f.write(str(trans + rot) + "\n")
    f.close()


if __name__ == "__main__":
    main()