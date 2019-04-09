import matplotlib
# matplotlib.use('Agg')
from ros.test import im_segment_single_frame, _extract_vertmap, vis_segmentations_vertmaps_detection
import timeit
import tensorflow as tf
import numpy as np
from datasets.factory import get_imdb
from generate_dataset.common import get_filename_prefix
import cv2
import os
from fcn.config import cfg, cfg_from_file, get_output_dir
import pprint
import random
import yaml
from utils.blob import pad_im, chromatic_transform, add_noise
from gt_synthesize_layer.minibatch import _generate_vertex_targets, _vis_minibatch
from transforms3d.quaternions import mat2quat, quat2mat
from generate_dataset.dope_to_posecnn import construct_posecnn_meta_data, get_dope_objects
import matplotlib.pyplot as plt
from ros.test import get_image_blob
from generate_dataset.dope_to_posecnn import linear_instance_segmentation_mask_image
from fcn.test import plot_data
import io


def setup():
    cfg_from_file("experiments/cfgs/lov_color_box.yml")
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
    print('Using config:')
    pprint.pprint(cfg)
    set_seed()

    imdb = get_imdb("lov_single_000_box_train")
    output_dir = get_output_dir(imdb, None)

    # K = np.array([[610.55994, 0, 306.86169], [0, 610.32086, 240.94547], [0, 0, 1]])
    K = np.array([[610., 0, 306.], [0, 610., 240.], [0, 0, 1]])
    meta_data = dict({'intrinsic_matrix': K, 'factor_depth': 10000.0})

    # This can not be imported at the top like it should be, but has to be imported after the default config has been merged with the file config
    from networks.factory import get_network
    network = get_network("vgg16_convs")

    saver = tf.train.Saver()
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=False, gpu_options=gpu_options))

    with open("generate_dataset/config.yaml", "r") as config:
        config_dict = yaml.load(config)

    model = config_dict["model"]
    print ('Loading model weights from {:s}').format(model)
    saver.restore(sess, model)

    return config_dict, sess, network, meta_data, imdb, output_dir


def subplot(imgs):
    for i, img in enumerate(imgs):
        plt.subplot(1, len(imgs), i+1)
        plt.imshow(img)
    plt.show()

def read_input_data(src_path_prefix):
    dope = True
    if dope:
        rgba = pad_im(cv2.imread(src_path_prefix + ".png", cv2.IMREAD_UNCHANGED), 16)
        if rgba.shape[2] == 4:
            im = np.copy(rgba[:, :, :3])
            alpha = rgba[:, :, 3]
            I = np.where(alpha == 0)
            im[I[0], I[1], :] = 0
        else:
            im = rgba

        depth_cv = cv2.imread(src_path_prefix + ".depth.png", cv2.IMREAD_ANYDEPTH)
        # chromatic transform
        if cfg.TRAIN.CHROMATIC:
            print("Chromatic transform")
            im = chromatic_transform(im)

        if cfg.TRAIN.ADD_NOISE:
            print("Adding noise")
            im = add_noise(im)
    else:
        im = cv2.imread(src_path_prefix + "-color.png")
        depth_cv = cv2.imread(src_path_prefix + "-depth.png", cv2.IMREAD_ANYDEPTH)
    return im, depth_cv


def read_label_data(src_path_prefix, intrinsic_matrix, num_classes, im_scales, extents, blob_height, blob_width):
    """ build the label blob """
    objects = get_dope_objects(src_path_prefix)
    meta_data = construct_posecnn_meta_data(objects, intrinsic_matrix)

    num_images = 1
    processed_depth = []
    processed_label = []
    processed_meta_data = []
    vertex_target_blob = np.zeros((num_images, blob_height, blob_width, 3 * num_classes), dtype=np.float32)
    vertex_weight_blob = np.zeros((num_images, blob_height, blob_width, 3 * num_classes), dtype=np.float32)
    pose_blob = np.zeros((0, 13), dtype=np.float32)

    gt_boxes = []

    for i in xrange(num_images):
        im_scale = im_scales[i]

        meta_data['cls_indexes'] = meta_data['cls_indexes'].flatten()
        if os.path.exists(src_path_prefix + ".depth.png"):
            im_depth = pad_im(cv2.imread(src_path_prefix + ".depth.png", cv2.IMREAD_UNCHANGED), 16)
        else:
            im_depth = np.zeros((blob_height, blob_width), dtype=np.float32)

        # read label image
        im = pad_im(cv2.imread(src_path_prefix + ".cs.png", cv2.IMREAD_UNCHANGED), 16)

        im = cv2.resize(im, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_NEAREST)

        # process annotation if training for two classes
        I = np.where(im == 1)
        im[:, :] = 0
        im[I[0], I[1]] = 1
        ind = np.where(meta_data['cls_indexes'] == 1)[0]
        cls_indexes_old = ind
        meta_data['cls_indexes'] = np.ones((len(ind),), dtype=np.float32)
        if len(meta_data['poses'].shape) == 2:
            meta_data['poses'] = np.reshape(meta_data['poses'], (3, 4, 1))
        meta_data['poses'] = meta_data['poses'][:, :, ind]
        meta_data['center'] = meta_data['center'][ind, :]

        im_labels = im.copy()
        processed_label.append(im_labels.astype(np.int32))

        # vertex regression targets and weights
        poses = meta_data['poses']
        if len(poses.shape) == 2:
            poses = np.reshape(poses, (3, 4, 1))

        vertmap = []

        center = meta_data['center']

        # check if mutiple same instances
        cls_indexes = meta_data['cls_indexes']
        if len(np.unique(cls_indexes)) < len(cls_indexes):
            is_multi_instances = 1
            # read mask image
            mask_img = cv2.imread(src_path_prefix + ".is.png", cv2.IMREAD_UNCHANGED)
            mask_img = linear_instance_segmentation_mask_image(objects, mask_img)
            try:
                # The mask image needs to be croped for simulation/dope data, because their masks are not black/white but are color masks.
                mask_img = mask_img[:, :, 0]
            except IndexError:
                pass
            mask = pad_im(mask_img, 16)
        else:
            is_multi_instances = 0
            mask = []

        vertex_target_blob[i, :, :, :], vertex_weight_blob[i, :, :, :] = \
            _generate_vertex_targets(im, meta_data['cls_indexes'], im_scale * center, poses, num_classes, vertmap, extents, \
                                     mask, is_multi_instances, cls_indexes_old, \
                                     vertex_target_blob[i, :, :, :], vertex_weight_blob[i, :, :, :])

        num = poses.shape[2]
        qt = np.zeros((num, 13), dtype=np.float32)
        for j in xrange(num):
            R = poses[:, :3, j]
            T = poses[:, 3, j]

            qt[j, 0] = i
            qt[j, 1] = meta_data['cls_indexes'][j]
            qt[j, 2:6] = 0  # fill box later
            qt[j, 6:10] = mat2quat(R)
            qt[j, 10:] = T

        pose_blob = np.concatenate((pose_blob, qt), axis=0)

        # construct the meta data
        """
        format of the meta_data
        intrinsic matrix: meta_data[0 ~ 8]
        inverse intrinsic matrix: meta_data[9 ~ 17]
        pose_world2live: meta_data[18 ~ 29]
        pose_live2world: meta_data[30 ~ 41]
        voxel step size: meta_data[42, 43, 44]
        voxel min value: meta_data[45, 46, 47]
        """
        K = np.matrix(meta_data['intrinsic_matrix']) * im_scale
        K[2, 2] = 1
        Kinv = np.linalg.pinv(K)
        mdata = np.zeros(48, dtype=np.float32)
        mdata[0:9] = K.flatten()
        mdata[9:18] = Kinv.flatten()
        processed_meta_data.append(mdata)

        # depth
        depth = im_depth.astype(np.float32, copy=True) / float(meta_data['factor_depth'])
        depth = cv2.resize(depth, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR)
        processed_depth.append(depth)

    # construct the blobs
    depth_blob = np.zeros((num_images, blob_height, blob_width, 1), dtype=np.float32)
    meta_data_blob = np.zeros((num_images, 1, 1, 48), dtype=np.float32)

    for i in xrange(num_images):
        depth_blob[i, :, :, 0] = processed_depth[i]
        meta_data_blob[i, 0, 0, :] = processed_meta_data[i]

    label_blob = np.zeros((num_images, blob_height, blob_width), dtype=np.int32)

    for i in xrange(num_images):
        label_blob[i, :, :] = processed_label[i]

    return depth_blob, label_blob, meta_data_blob, vertex_target_blob, vertex_weight_blob, pose_blob, gt_boxes


def get_plot(sess, data, im_label, imdb, vertmap, labels, rois, poses, intrinsic_matrix, output_dir):
    plot_data(data, None, im_label, imdb._class_colors, vertmap, labels, rois, poses, [], intrinsic_matrix, imdb.num_classes, imdb._classes, imdb._points_all)
    img_str_placeholder = tf.placeholder(tf.string)
    image = tf.image.decode_png(img_str_placeholder, channels=4)
    # Add the batch dimension
    image_expanded = tf.expand_dims(image, 0)

    # Add image summary
    img_op = tf.summary.image("Val predictions", image_expanded)
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=600)
    buf.seek(0)
    img_summary = sess.run(img_op, feed_dict={img_str_placeholder: buf.getvalue()})
    val_writer = tf.summary.FileWriter(output_dir + "/val", sess.graph)
    val_writer.add_summary(img_summary, 1)


def test_weights(sess):
    print("Testing vars")
    for var in tf.global_variables():
        network_var = sess.run(var)
        loaded = np.load(os.path.join("/home/satco/kaju", var.name.replace("/", "_").replace(":", "_")) + ".npy")
        is_close = np.isclose(network_var, loaded, 0.00001)
        if np.isin(False, is_close):
            print(var.name)
            print(is_close)
        else:
            print("Passed: ", var.name)


def set_seed():
    randomize = False
    if not randomize:
        # fix the random seeds (numpy and caffe) for reproducibility
        seed = 30
    else:
        seed = random.randint(0, 100)
    print("Using seed: ", seed)
    tf.set_random_seed(seed)
    np.random.seed(seed)


def main():
    print("Interactive mode: ", matplotlib.is_interactive())
    print("Backend: ", matplotlib.get_backend())
    config_dict, sess, network, meta_data, imdb, output_dir = setup()
    # test_weights(sess)

    data_folder = config_dict["data_folder"]
    start = config_dict["start"]
    end = config_dict["end"]

    for i in range(1):
        j = random.randint(start, end)
        j = 3090
        prefix = get_filename_prefix(j)
        print("Prefix: ", prefix)
        src_path_prefix = os.path.join(data_folder, prefix)
        im, depth_cv = read_input_data(src_path_prefix)
        # compute image blob
        im_blob, im_depth_blob, im_normal_blob, im_scale_factors, height, width = get_image_blob(im, depth_cv, meta_data, cfg)
        depth_blob, label_blob, meta_data_blob, vertex_target_blob, vertex_weight_blob, pose_blob, gt_boxes = \
            read_label_data(src_path_prefix, meta_data['intrinsic_matrix'], 2, [1], imdb._extents, height, width)
        # run network
        start_time = timeit.default_timer()
        # _vis_minibatch(im_blob, im_depth_blob, depth_blob, label_blob, meta_data_blob, vertex_target_blob, pose_blob, imdb._extents, imdb._points_all, imdb._class_colors)
        data, labels, probs, vertex_pred, rois, poses = im_segment_single_frame(sess, network, im_blob, im_depth_blob, im_normal_blob, meta_data, imdb._extents, imdb._points_all, imdb._symmetry, imdb.num_classes,
                                                                          cfg, output_dir, i, depth_blob, label_blob, meta_data_blob, vertex_target_blob, vertex_weight_blob, pose_blob, gt_boxes)
        elapsed = timeit.default_timer() - start_time
        print("ELAPSED TIME:")
        print(elapsed)
        poses_icp = []

        im_label = imdb.labels_to_image(data, labels)

        if cfg.TEST.VISUALIZE:
            vertmap = _extract_vertmap(labels, vertex_pred, imdb._extents, imdb.num_classes)
            # get_plot(sess, data, im_label, imdb, vertmap, labels, rois, poses, meta_data['intrinsic_matrix'], output_dir)
            vis_segmentations_vertmaps_detection(data, depth_cv, im_label, imdb._class_colors, vertmap, labels, rois, poses, poses_icp, meta_data['intrinsic_matrix'],
                                                 imdb.num_classes, imdb._classes, imdb._points_all)


if __name__ == '__main__':
    main()