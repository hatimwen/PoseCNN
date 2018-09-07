import cv2
import rosbag
import rospy
from cv_bridge import CvBridge


def main():
    bag = rosbag.Bag("/home/satco/PycharmProjects/PoseCNN/bag/dataset_one_box.bag")
    topics = ["/tf",

              "/camera1/depth/camera_info", "/camera1/depth/image_rect_raw", "/camera1/color/camera_info",
              "/camera1/color/image_raw", "/camera1/aligned_depth_to_color/camera_info",
              "/camera1/aligned_depth_to_color/image_raw",

              "/camera2/depth/camera_info", "/camera2/depth/image_rect_raw", "/camera2/color/camera_info",
              "/camera2/color/image_raw", "/camera2/aligned_depth_to_color/camera_info",
              "/camera2/aligned_depth_to_color/image_raw"]
    counter = 0
    bridge = CvBridge()
    for topic, msg, t in bag.read_messages(topics=topics, start_time=rospy.Time(1535537364, 810703)):
        if topic == "/camera1/color/image_raw":
            print("Showing image " + str(counter))
            image = bridge.imgmsg_to_cv2(msg, "bgr8")
            mask_name = "data/images/cube" + str(counter) + ".png"
            mask = cv2.imread(mask_name)
            # cv2.imshow("Image", image)
            # cv2.imshow("Mask", mask)
            alpha = 0.2
            image_with_mask = cv2.addWeighted(mask, alpha, image, 1 - alpha, 0)
            cv2.imshow("Image with mask", image_with_mask)
            cv2.waitKey(50)
            counter += 1


if __name__ == "__main__":
    main()