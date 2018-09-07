import rosbag
import rospy


def extract_box_position(bag, measurement_time_secs, measurement_time_nsecs, topics):
    for topic, msg, t in bag.read_messages(topics=topics):

        if topic == "/tf":
            tf_msg = msg.transforms[0]
            # print(tf_msg.child_frame_id)
            if tf_msg.child_frame_id == "box_corner_tracker":
                secs = tf_msg.header.stamp.secs
                nsecs = tf_msg.header.stamp.nsecs
                # print(secs)
                # print(nsecs)
                if secs == measurement_time_secs and nsecs == measurement_time_nsecs:
                    print(msg)
                    print("Found transform")
                    return tf_msg


# def get_points_to_backproject_in_box_corner_tracker_frame(box_dimensions):


# def apply_transform(msg):
#     br = tf.TransformBroadcaster()
#     br.sendTransform((0.0, 2.0, 0.0), (0.0, 0.0, 0.0, 1.0), rospy.Time.now(), "camera1", "davis")
#     tf.listener.TransformListener.lookupTransform("vicon", "camera1", rospy.Time(0))


def print_timestamp(timestamp):
    print(str(timestamp.secs) + "." + str(timestamp.nsecs))


def tf_msg_to_dict(tf_msg):
    # translation = (-0.029793949479807992, -0.00576132479343137, 0.11578614371327833)
    # rotation = (0.20107065010341968, -0.1307783221491868, 0.56686050120762888, -0.78812232317925601)
    # tf.TransformerROS.fromTranslationRotation(translation, rotation)
    x = tf_msg.transform.translation.x
    y = tf_msg.transform.translation.y
    z = tf_msg.transform.translation.z
    xr = tf_msg.transform.rotation.x
    yr = tf_msg.transform.rotation.y
    zr = tf_msg.transform.rotation.z
    w = tf_msg.transform.rotation.w
    return [x, y, z, xr, yr, zr, w]


def main():
    bag = rosbag.Bag("/home/satco/PycharmProjects/PoseCNN/bag/dataset_one_box.bag")
    box_dimensions = [34.9, 21.3, 12.4]
    topics = ["/tf",

              "/camera1/depth/camera_info", "/camera1/depth/image_rect_raw", "/camera1/color/camera_info",
              "/camera1/color/image_raw", "/camera1/aligned_depth_to_color/camera_info",
              "/camera1/aligned_depth_to_color/image_raw",

              "/camera2/depth/camera_info", "/camera2/depth/image_rect_raw", "/camera2/color/camera_info",
              "/camera2/color/image_raw", "/camera2/aligned_depth_to_color/camera_info",
              "/camera2/aligned_depth_to_color/image_raw"]
    box_position_msg = extract_box_position(bag, 1535537356, 262816, topics)
    print(type(box_position_msg.transform))
    box_position = tf_msg_to_dict(box_position_msg)
    with open("data/box_positions.txt", "w") as f:
        f.write(str(box_position) + "\n")
    # bridge = CvBridge()
    f = open("data/camera1_positions.txt", "w")
    # f2 = open("data/timestamps2.txt", "w")
    timestamps = []
    timestamps2 = []
    for topic, msg, t in bag.read_messages(topics=topics, start_time=rospy.Time(1535537364, 810703)):
        if topic == "/camera1/color/image_raw":
            timestamps.append(msg.header.stamp)
        if topic == "/camera2/color/image_raw":
            timestamps2.append(msg.header.stamp)
    print("Found " + str(len(timestamps)) + "camera1 timestamps")
    print("Found " + str(len(timestamps2)) + "camera2 timestamps")
    last_tf_msg = None
    tf_timestamp = None
    counter = 0
    stamp_counter = 1
    for topic, msg, t in bag.read_messages(topics=topics):
        if topic == "/tf":
            tf_msg = msg.transforms[0]
            if tf_msg.child_frame_id == "davis":
                if counter == 0:
                    counter = 1
                    last_tf_msg = tf_msg
                    tf_timestamp = tf_msg.header.stamp
                    continue
                last_tf_timestamp = tf_timestamp
                tf_timestamp = tf_msg.header.stamp
                # this was initially a if condition but had to be changed to a while loop, because sometimes there are
                # big(0.12s) gaps in the tf timestamps of the davis frame
                while stamp_counter < len(timestamps) and last_tf_timestamp < timestamps[stamp_counter] < tf_timestamp:
                    if abs(timestamps[stamp_counter] - last_tf_timestamp) < abs(timestamps[stamp_counter] - tf_timestamp):
                        camera_position = tf_msg_to_dict(last_tf_msg)
                        f.write(str(camera_position) + "\n")
                    else:
                        camera_position = tf_msg_to_dict(tf_msg)
                        f.write(str(camera_position) + "\n")
                    stamp_counter += 1

    f.close()


if __name__ == "__main__":
    main()