#!/usr/bin/env python

# --------------------------------------------------------
# FCN
# Copyright (c) 2016 RSE at UW
# Licensed under The MIT License [see LICENSE for details]
# Written by Yu Xiang
# --------------------------------------------------------

"""Test a FCN on an image database."""

import _init_paths
from fcn.config import cfg, cfg_from_file
from datasets.factory import get_imdb
import argparse
import pprint
import time, os, sys
import tensorflow as tf
import os.path as osp
import numpy as np
import rospy
import rosbag
from cv_bridge import CvBridge, CvBridgeError
from test import test_ros
import faulthandler
from generate_dataset.export_data_from_ros_bag import read_dataset_times
import matplotlib.pyplot as plt
from tools.test_folder import setup

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Test a Fast R-CNN network')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU id to use',
                        default=0, type=int)
    parser.add_argument('--weights', dest='pretrained_model',
                        help='pretrained model',
                        default=None, type=str)
    parser.add_argument('--model', dest='model',
                        help='model to test',
                        default=None, type=str)
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file', default=None, type=str)
    parser.add_argument('--wait', dest='wait',
                        help='wait until net file exists',
                        default=True, type=bool)
    parser.add_argument('--imdb', dest='imdb_name',
                        help='dataset to test',
                        default='shapenet_scene_val', type=str)
    parser.add_argument('--network', dest='network_name',
                        help='name of the network',
                        default=None, type=str)
    parser.add_argument('--rig', dest='rig_name',
                        help='name of the camera rig file',
                        default=None, type=str)
    parser.add_argument('--cad', dest='cad_name',
                        help='name of the CAD file',
                        default=None, type=str)
    parser.add_argument('--kfusion', dest='kfusion',
                        help='run kinect fusion or not',
                        default=False, type=bool)
    parser.add_argument('--pose', dest='pose_name',
                        help='name of the pose files',
                        default=None, type=str)
    parser.add_argument('--background', dest='background_name',
                        help='name of the background file',
                        default=None, type=str)
    parser.add_argument('--bag', dest='bag_name',
                        help='name of the bag file',
                        default=None, type=str)
    parser.add_argument('--color_topic', dest='color_topic',
                        help='name of the color topic',
                        default="/camera/color/image_raw", type=str)
    parser.add_argument('--depth_topic', dest='depth_topic',
                        help='name of the depth topic',
                        default="/camera/aligned_depth_to_color/image_raw", type=str)
                        # default="/camera/depth/image_rect_raw", type=str)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    config_dict, sess, network, meta_data, imdb = setup()

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

    color_topic = "/camera/color/image_raw"
    depth_topic = "/camera/aligned_depth_to_color/image_raw"

    try:
        start_time, end_time, times = read_dataset_times(dataset_name, "generate_dataset/")
    except IOError:
        start_time = rospy.Time(bag.get_start_time())
        end_time = rospy.Time(bag.get_end_time())
    for topic, msg, t in bag.read_messages(topics=[color_topic, depth_topic], start_time=start_time, end_time=end_time):
        print count, topic, type(msg)
        if topic == color_topic:
            rgb = msg
        if depth_topic in topics:
            if topic == depth_topic:
                depth = msg

        if count % skip_frames == 0 or (count - 1) % skip_frames == 0:
            count_inner += 1
            if count_inner % 2 == 0:
                try:
                    test_ros(sess, network, imdb, meta_data, cfg, rgb, depth, cv_bridge, count/2 - 1)
                except NameError:
                    test_ros(sess, network, imdb, meta_data, cfg, rgb, None, cv_bridge, count/2 - 1)

        count += 1

    bag.close()
