from ros.test import im_segment_single_frame, _extract_vertmap, vis_segmentations_vertmaps_detection
import timeit
import tensorflow as tf
import numpy as np
from datasets.factory import get_imdb
from generate_dataset.common import get_filename_prefix
import cv2
import os
from fcn.config import cfg, cfg_from_file
import pprint
import random


def prepare_config():
    cfg.GPU_ID = 0
    device_name = '/gpu:{:d}'.format(0)
    print device_name

    cfg.TRAIN.NUM_STEPS = 1
    cfg.TRAIN.GRID_SIZE = cfg.TEST.GRID_SIZE
    cfg.TRAIN.TRAINABLE = False

    cfg.RIG = "data/LOV/camera.json"
    cfg.CAD = "data/LOV/models.txt"
    cfg.POSE = "data/LOV/poses.txt"
    cfg.BACKGROUND = "data/cache/backgrounds.pkl"
    cfg.IS_TRAIN = False


def main():
    cfg_from_file("experiments/cfgs/lov_color_box.yml")
    prepare_config()
    print('Using config:')
    pprint.pprint(cfg)

    randomize = True
    if not randomize:
        # fix the random seeds (numpy and caffe) for reproducibility
        tf.set_random_seed(cfg.RNG_SEED)
        np.random.seed(cfg.RNG_SEED)

    imdb = get_imdb("lov_single_000_box_train")

    K = np.array([[610.55994, 0, 306.86169], [0, 610.32086, 240.94547], [0, 0, 1]])
    meta_data = dict({'intrinsic_matrix': K, 'factor_depth': 10000.0})

    # This can not be imported at the top like it should be, but has to be imported after the default config has been merged with the file config
    from networks.factory import get_network
    network = get_network("vgg16_convs")

    saver = tf.train.Saver()
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options))
    # model = "trained_nets/run219/vgg16_fcn_color_single_frame_2d_pose_add_sym_lov_box_iter_397_epoch_7.ckpt"
    # model = "trained_nets/run170/vgg16_fcn_color_single_frame_2d_pose_add_sym_lov_box_iter_497_epoch_4.ckpt"
    model = "trained_nets/run275/vgg16_fcn_color_single_frame_2d_pose_add_sym_lov_box_iter_397_epoch_4.ckpt"
    saver.restore(sess, model)
    print ('Loading model weights from {:s}').format(model)

    data_folder = "/home/satco/catkin_ws/src/Deep_Object_Pose/data/Static"

    for i in range(5):
        j = random.randint(4000, 4999)
        prefix = get_filename_prefix(j)
        im = cv2.imread(os.path.join(data_folder, prefix + ".png"))
        depth_cv = cv2.imread(os.path.join(data_folder, prefix + ".depth.png"), cv2.IMREAD_ANYDEPTH)
        # run network
        start_time = timeit.default_timer()
        labels, probs, vertex_pred, rois, poses = im_segment_single_frame(sess, network, im, depth_cv, meta_data, imdb._extents, imdb._points_all, imdb._symmetry, imdb.num_classes,
                                                                          cfg)
        elapsed = timeit.default_timer() - start_time
        print("ELAPSED TIME:")
        print(elapsed)
        poses_icp = []

        im_label = imdb.labels_to_image(im, labels)

        if cfg.TEST.VISUALIZE:
            vertmap = _extract_vertmap(labels, vertex_pred, imdb._extents, imdb.num_classes)
            vis_segmentations_vertmaps_detection(im, depth_cv, im_label, imdb._class_colors, vertmap, labels, rois, poses, poses_icp, meta_data['intrinsic_matrix'],
                                                 imdb.num_classes, imdb._classes, imdb._points_all)


if __name__ == '__main__':
    main()