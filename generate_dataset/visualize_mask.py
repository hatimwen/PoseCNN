import cv2
import rosbag
import rospy
from cv_bridge import CvBridge


def main():
    bag = rosbag.Bag("/home/satco/PycharmProjects/PoseCNN/bag/dataset_one_box.bag")
    # bag = rosbag.Bag("/home/satco/PycharmProjects/PoseCNN/bag/test.bag")
    # topics = ["/camera1/color/image_raw", "/camera2/color/image_raw"]
    topics = ["/camera/color/image_raw"]
    # counter = -20
    counter = 0
    bridge = CvBridge()
    for topic, msg, t in bag.read_messages(topics=topics, start_time=rospy.Time(1537799716, 30952)):
        print(msg.header.stamp)
        # if topic == "/camera1/color/image_raw":
        if topic == "/camera/color/image_raw":
            # print(msg.header.stamp)
            if counter < 0:
                counter += 1
                continue
            # print("Showing image " + str(counter))
            image = bridge.imgmsg_to_cv2(msg, "bgr8")
            mask_name = "data/images/cube" + str(counter) + ".png"
            mask = cv2.imread(mask_name)
            alpha = 0.5
            image_with_mask = cv2.addWeighted(mask, alpha, image, 1 - alpha, 0)
            cv2.imshow("Image with mask", image_with_mask)
            cv2.waitKey(300)
            counter += 1


if __name__ == "__main__":
    main()