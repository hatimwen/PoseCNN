import rosbag
from cv_bridge import CvBridge, CvBridgeError
import inspect


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
                    print("Found transform")
                    return tf_msg


# def get_points_to_backproject_in_box_corner_tracker_frame(box_dimensions):



def main():
    bag = rosbag.Bag("/home/satco/catkin_ws/src/thesis/bag/dataset_one_box.bag")
    box_dimensions = [34.9, 21.3, 12.4]
    topics = ["/tf",

              "/camera1/depth/camera_info", "/camera1/depth/image_rect_raw", "/camera1/color/camera_info",
              "/camera1/color/image_raw", "/camera1/aligned_depth_to_color/camera_info",
              "/camera1/aligned_depth_to_color/image_raw",

              "/camera2/depth/camera_info", "/camera2/depth/image_rect_raw", "/camera2/color/camera_info",
              "/camera2/color/image_raw", "/camera2/aligned_depth_to_color/camera_info",
              "/camera2/aligned_depth_to_color/image_raw"]
    box_position_msg = extract_box_position(bag, 1535537356, 262816, topics)
    bridge = CvBridge()
    print(box_position_msg)
    for topic, msg, t in bag.read_messages(topics=topics):
        cv_image = bridge.imgmsg_to_cv2(image_message, desired_encoding="passthrough")
        print(topic)
        # if topic == "/tf":
        #     print(topic, msg, t)
        # break


if __name__ == "__main__":
    main()