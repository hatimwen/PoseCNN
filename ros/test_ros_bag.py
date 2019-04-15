#!/usr/bin/env python

# --------------------------------------------------------
# FCN
# Copyright (c) 2016 RSE at UW
# Licensed under The MIT License [see LICENSE for details]
# Written by Yu Xiang
# --------------------------------------------------------

"""Test a FCN on an image database."""

import timeit
from fcn.config import cfg
import numpy as np
import rospy
import rosbag
from cv_bridge import CvBridge
from ros.test import get_image_blob, _extract_vertmap, vis_segmentations_vertmaps_detection
from generate_dataset.export_data_from_ros_bag import read_dataset_times
from tools.test_dataset import setup, queue_up_data, forward_pass
from generate_dataset.common import get_meta_data
import os


def get_data(img, depth):
    if depth:
        if depth.encoding == '32FC1':
            depth = cv_bridge.imgmsg_to_cv2(depth) * 1000
            depth = np.array(depth, dtype=np.uint16)
        elif depth.encoding == '16UC1':
            depth = cv_bridge.imgmsg_to_cv2(depth)
        else:
            rospy.logerr_throttle(1, 'Unsupported depth type. Expected 16UC1 or 32FC1, got {}'.format(depth.encoding))

    # image
    if img:
        img = cv_bridge.imgmsg_to_cv2(img, 'bgr8')
    return img, depth


def create_empty_label_data(meta_data, num_classes, im_depth):
    K = np.matrix(meta_data['intrinsic_matrix'])
    K[2, 2] = 1
    Kinv = np.linalg.pinv(K)
    mdata = np.zeros(48, dtype=np.float32)
    mdata[0:9] = K.flatten()
    mdata[9:18] = Kinv.flatten()
    meta_data_blob = np.zeros((1, 1, 1, 48), dtype=np.float32)
    meta_data_blob[0, 0, 0, :] = mdata

    # use a fake label blob of ones
    try:
        height = int(im_depth.shape[0])
    except AttributeError:
        print(type(im_depth))
        print(im_depth)
    width = int(im_depth.shape[1])

    label_blob = np.ones((1, height, width), dtype=np.int32)

    pose_blob = np.zeros((1, 13), dtype=np.float32)
    vertex_target_blob = np.zeros((1, height, width, 3 * num_classes), dtype=np.float32)
    vertex_weight_blob = np.zeros((1, height, width, 3 * num_classes), dtype=np.float32)
    return label_blob, pose_blob, vertex_target_blob, vertex_weight_blob, meta_data_blob


if __name__ == '__main__':
    config_dict, sess, network, imdb, output_dir = setup()
    meta_data = get_meta_data()

    bag_path = os.path.join(config_dict["bags_path"], config_dict["test_bag"] + ".bag")

    # ros bag
    bag = rosbag.Bag(bag_path)
    cv_bridge = CvBridge()

    count = 0
    count_inner = 0
    dataset_name = config_dict["test_bag"]

    topics = bag.get_type_and_topic_info().topics

    rgb = None
    depth = None

    skip_frames = config_dict["skip_frames"]

    color_topic = config_dict["color_topic"]
    depth_topic = config_dict["depth_topic"]

    try:
        start_time, end_time, times = read_dataset_times(dataset_name, "generate_dataset/")
    except IOError:
        start_time = rospy.Time(bag.get_start_time())
        end_time = rospy.Time(bag.get_end_time())
    for topic, msg, t in bag.read_messages(topics=[color_topic, depth_topic], start_time=start_time, end_time=end_time):
        if topic == color_topic:
            img = msg
        if depth_topic in topics:
            if topic == depth_topic:
                depth = msg

        if count % skip_frames == 0 or (count - 1) % skip_frames == 0:
            count_inner += 1
            if count_inner % 2 == 0:
                img, depth = get_data(img, depth)
                im_blob, im_depth_blob, im_normal_blob, im_scale_factors, height, width = get_image_blob(img, depth, meta_data, cfg)
                label_blob, pose_blob, vertex_target_blob, vertex_weight_blob, meta_data_blob = create_empty_label_data(meta_data, 2, depth)
                start_time = timeit.default_timer()
                queue_up_data(sess, im_blob, network, label_blob, vertex_target_blob, vertex_weight_blob, meta_data_blob, imdb._extents, imdb._points_all, imdb._symmetry,
                              pose_blob)
                data, labels, probs, vertex_pred, rois, poses, losses_values = forward_pass(sess, network)
                elapsed = timeit.default_timer() - start_time
                if cfg.TEST.VISUALIZE:
                    vertmap = _extract_vertmap(labels, vertex_pred, imdb._extents, imdb.num_classes)
                    im_label = imdb.labels_to_image(data, labels)
                    poses_icp = []
                    vis_segmentations_vertmaps_detection(data, depth, im_label, imdb._class_colors, vertmap, labels, rois, poses, poses_icp, meta_data['intrinsic_matrix'],
                                                         imdb.num_classes, imdb._classes, imdb._points_all)

        count += 1

    bag.close()
