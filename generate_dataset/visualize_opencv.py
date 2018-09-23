import cv2
from cv_bridge import CvBridge
from export_tf import fill_transformer
import numpy as np
from pyquaternion import Quaternion
import rosbag
import rospy


def ros_to_blender_quat(qaut):
    return qaut[-1], qaut[0], qaut[1], qaut[2]


def get_corner(tf_t):
    corner = []
    for i in range(1, 9):
        (trans, rot) = tf_t.lookupTransform("vicon", "box_corner" + str(i), rospy.Time(1536832337, 57485))
        corner.append(list(trans))
    corners = np.float32(corner)
    return corners


def calculate_point(K, R, t, corner):
    print("K: " + str(K))
    print("R: " + str(R))
    print("T: " + str(t))
    print("Corner: " + str(corner))
    R = np.hstack((R, t))
    print("R: " + str(R))
    result = np.dot(np.dot(K, R), corner)
    print("By hand: " + str(result))


def main():
    bag = rosbag.Bag("/home/satco/PycharmProjects/PoseCNN/bag/test.bag")
    topics = ["/camera/color/image_raw"]
    tf_t = fill_transformer(bag)
    corner_input = get_corner(tf_t)
    print(corner_input)
    # print(corner_input)
    camera_matrix = np.float32([[610.55992534, 0, 306.86169342], [0, 610.32086262, 240.94547232], [0, 0, 1]])
    dist = np.float32([[0.10793695], [-0.21546604], [0.00045875], [-0.00670819]])
    dist = np.float32([[0], [0], [0], [0]])
    # height = 480
    # width = 640
    counter = 0
    bridge = CvBridge()
    for topic, msg, t in bag.read_messages(topics=topics, start_time=rospy.Time(1536832352, 412808)):
        # print(msg.header.stamp)
        if topic == "/camera/color/image_raw":
            (trans, rot) = tf_t.lookupTransform("vicon", "camera", msg.header.stamp)
            trans = np.float32([[trans[0]], [trans[1]], [trans[2]]])
            print("Rot: " + str(rot))
            # R = camera_quat.rotation_matrix.transpose()
            camera_quat = Quaternion(ros_to_blender_quat(rot))
            R = camera_quat.rotation_matrix
            camera_rodrigues, jacobian = cv2.Rodrigues(R)
            # print(camera_quat.rotation_matrix)
            calculate_point(camera_matrix, R, trans, np.float32([[0.7136561], [-0.18407732], [0.15049139], [1]]))
            print(camera_rodrigues)
            if counter < 0:
                counter += 1
                continue
            image = bridge.imgmsg_to_cv2(msg, "bgr8")
            corners, jacobian = cv2.projectPoints(corner_input, camera_rodrigues, trans, camera_matrix, dist)
            # corners, jacobian = cv2.projectPoints(corner_input, camera_rodrigues, trans, camera_matrix)
            for corner in corners:
                corner = tuple(corner[0])
                print(corner)
                if corner[0] > 0 and corner[1] > 0:
                    cv2.circle(image, corner, 63, (0, 0, 255), -1)
            # print(imagePoints)
            cv2.circle(image, (50, 50), 5, (255, 0, 0), -1)
            cv2.imshow("Image", image)
            cv2.waitKey(100)
            counter += 1


if __name__ == "__main__":
    main()
