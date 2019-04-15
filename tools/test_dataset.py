from ros.test import _extract_vertmap, vis_segmentations_vertmaps_detection
import timeit
import tensorflow as tf
import numpy as np
from datasets.factory import get_imdb
from generate_dataset.common import get_filename_prefix, get_intrinsic_matrix
import cv2
import os
from fcn.config import cfg, cfg_from_file, get_output_dir
import pprint
import random
import yaml
from utils.blob import pad_im
from gt_synthesize_layer.minibatch import _generate_vertex_targets, _vis_minibatch
from transforms3d.quaternions import mat2quat
from generate_dataset.dope_to_posecnn import construct_posecnn_meta_data, get_dope_objects
import matplotlib.pyplot as plt
from ros.test import get_image_blob
from generate_dataset.dope_to_posecnn import linear_instance_segmentation_mask_image
from fcn.test import plot_data
import io
from tools.common import smooth_l1_loss_vertex, combine_poses
import scipy.io


def setup():
    cfg_from_file("experiments/cfgs/lov_color_box.yml")
    cfg.GPU_ID = 0
    device_name = '/gpu:{:d}'.format(0)
    print(device_name)

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

    # This can not be imported at the top like it should be, but has to be imported after the default config has been merged with the file config
    from networks.factory import get_network
    network = get_network("vgg16_convs")

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=False, gpu_options=gpu_options))

    with open("generate_dataset/config.yaml", "r") as config:
        config_dict = yaml.load(config)

    model = config_dict["model"]
    print ('Loading model weights from {:s}').format(model)
    # saver = tf.train.import_meta_graph(model + ".meta")
    saver = tf.train.Saver()
    saver.restore(sess, model)

    return config_dict, sess, network, imdb, output_dir


def subplot(imgs):
    for i, img in enumerate(imgs):
        plt.subplot(1, len(imgs), i+1)
        plt.imshow(img)
    plt.show()


def get_meta_data(dope, src_path_prefix, intrinsic_matrix):
    if dope:
        objects = get_dope_objects(src_path_prefix)
        meta_data = construct_posecnn_meta_data(objects, intrinsic_matrix)
    else:
        objects = None
        meta_data = scipy.io.loadmat(src_path_prefix + "-meta.mat")
    return objects, meta_data


def get_postfixes(dope):
    if dope:
        color = ".png"
        depth = ".depth.png"
        cls = ".cs.png"
        instance = ".is.png"
    else:
        color = "-color.png"
        depth = "-depth.png"
        cls = "-label.png"
        instance = "-mask.png"
    return color, depth, cls, instance


def read_input_data(src_path_prefix, color, depth):
    rgba = pad_im(cv2.imread(src_path_prefix + color, cv2.IMREAD_UNCHANGED), 16)
    if rgba.shape[2] == 4:
        im = np.copy(rgba[:, :, :3])
        alpha = rgba[:, :, 3]
        I = np.where(alpha == 0)
        im[I[0], I[1], :] = 0
    else:
        im = rgba

    depth_cv = cv2.imread(src_path_prefix + depth, cv2.IMREAD_ANYDEPTH)
    return im, depth_cv


def read_label_data(src_path_prefix, meta_data, num_classes, im_scales, extents, blob_height, blob_width, depth, cls, instance, objects):
    """ build the label blob """
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
        if os.path.exists(src_path_prefix + depth):
            im_depth = pad_im(cv2.imread(src_path_prefix + depth, cv2.IMREAD_UNCHANGED), 16)
        else:
            im_depth = np.zeros((blob_height, blob_width), dtype=np.float32)

        # read label image
        im = pad_im(cv2.imread(src_path_prefix + cls, cv2.IMREAD_UNCHANGED), 16)

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
            mask_img = cv2.imread(src_path_prefix + instance, cv2.IMREAD_UNCHANGED)
            if objects:
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

        vertex_target_blob[i, :, :, :], vertex_weight_blob[i, :, :, :] = _generate_vertex_targets(im, meta_data['cls_indexes'], im_scale * center, poses, num_classes, vertmap,
                                                                                                  extents, mask, is_multi_instances, cls_indexes_old,
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


def build_loses(net):
    loss_regu = tf.add_n(tf.losses.get_regularization_losses(), 'regu')

    loss_cls = net.get_output('loss_cls')

    vertex_pred = net.get_output('vertex_pred')

    vertex_targets = net.get_output('vertex_targets')

    vertex_weights = net.get_output('vertex_weights')

    loss_vertex = 3 * smooth_l1_loss_vertex(vertex_pred, vertex_targets, vertex_weights)

    loss_pose = net.get_output('loss_pose')[0]

    loss = loss_cls + loss_vertex + loss_pose + loss_regu

    losses = {
        "regu": loss_regu,
        "loss": loss,
        "cls": loss_cls,
        "vertex": loss_vertex,
        "pose": loss_pose
    }

    loss_placeholder = tf.placeholder(tf.float32, shape=())
    loss_cls_placeholder = tf.placeholder(tf.float32, shape=())
    loss_vertex_placeholder = tf.placeholder(tf.float32, shape=())
    loss_pose_placeholder = tf.placeholder(tf.float32, shape=())

    loss_val_op = tf.summary.scalar('loss_val', loss_placeholder)
    loss_cls_val_op = tf.summary.scalar('loss_cls_val', loss_cls_placeholder)
    loss_vertex_val_op = tf.summary.scalar('loss_vertex_val', loss_vertex_placeholder)
    loss_pose_val_op = tf.summary.scalar('loss_pose_val', loss_pose_placeholder)
    losses_ops = {
        "loss": loss_val_op,
        "cls": loss_cls_val_op,
        "vertex": loss_vertex_val_op,
        "pose": loss_pose_val_op
    }

    placeholders = [loss_placeholder, loss_cls_placeholder, loss_vertex_placeholder, loss_pose_placeholder]

    return losses, losses_ops, placeholders


def queue_up_data(sess, im_blob, net, label_blob, vertex_target_blob, vertex_weight_blob, meta_data_blob, extents, points, symmetry, pose_blob):
    keep_prob = 1.0
    is_train = False

    feed_dict = {net.data: im_blob, net.gt_label_2d: label_blob, net.keep_prob: keep_prob, net.is_train: is_train, net.vertex_targets: vertex_target_blob,
                 net.vertex_weights: vertex_weight_blob, net.meta_data: meta_data_blob, net.extents: extents, net.points: points, net.symmetry: symmetry, net.poses: pose_blob}

    sess.run(net.enqueue_op, feed_dict=feed_dict)


def forward_pass(sess, net, losses=None):
    if losses:
        loss = losses["loss"]
        loss_cls = losses["cls"]
        loss_vertex = losses["vertex"]
        loss_pose = losses["pose"]

        data, labels_2d, probs, vertex_pred, rois, poses_init, poses_pred, loss_value, loss_cls_value, loss_vertex_value, loss_pose_value = \
            sess.run([net.get_output("data"), net.get_output('label_2d'), net.get_output('prob_normalized'), net.get_output('vertex_pred'), net.get_output('rois'),
                      net.get_output('poses_init'), net.get_output('poses_tanh'), loss, loss_cls, loss_vertex, loss_pose])
        losses_values = [loss_value[0], loss_cls_value, loss_vertex_value, loss_pose_value[0]]
    else:
        data, labels_2d, probs, vertex_pred, rois, poses_init, poses_pred = \
            sess.run([net.get_output("data"), net.get_output('label_2d'), net.get_output('prob_normalized'), net.get_output('vertex_pred'), net.get_output('rois'),
                      net.get_output('poses_init'), net.get_output('poses_tanh')])
        losses_values = []

    return combine_poses(data, rois, poses_init, poses_pred, probs, vertex_pred, labels_2d, losses_values)


def main():
    config_dict, sess, network, imdb, output_dir = setup()
    intrinsic_matrix = get_intrinsic_matrix()

    data_folder = config_dict["data_folder"]
    skip_frames = config_dict["skip_frames"]
    files_to_test_name = config_dict["files_to_test_name"]
    dope = True
    color, depth, cls, instance = get_postfixes(dope)

    losses, losses_ops, placeholders = build_loses(network)
    train_writer = tf.summary.FileWriter(output_dir + "/test", sess.graph)
    losses_val = []
    losses_cls_val = []
    losses_vertex_val = []
    losses_pose_val = []
    files_to_test = open(files_to_test_name)
    indexes = [int(file.split("/")[1]) for file in files_to_test]
    for i in range(0, len(indexes), skip_frames):
        j = indexes[i]
        if i % 8 == 0:
            losses_val = []
            losses_cls_val = []
            losses_vertex_val = []
            losses_pose_val = []
        prefix = get_filename_prefix(j)
        src_path_prefix = os.path.join(data_folder, prefix)
        objects, meta_data = get_meta_data(dope, src_path_prefix, intrinsic_matrix)

        im, depth_cv = read_input_data(src_path_prefix, color, depth)
        # compute image blob
        im_blob, im_depth_blob, im_normal_blob, im_scale_factors, height, width = get_image_blob(im, depth_cv, meta_data, cfg)
        depth_blob, label_blob, meta_data_blob, vertex_target_blob, vertex_weight_blob, pose_blob, gt_boxes = \
            read_label_data(src_path_prefix, meta_data, 2, [1], imdb._extents, height, width, depth, cls, instance, objects)
        # run network
        start_time = timeit.default_timer()
        queue_up_data(sess, im_blob, network, label_blob, vertex_target_blob, vertex_weight_blob, meta_data_blob, imdb._extents, imdb._points_all, imdb._symmetry, pose_blob)
        _vis_minibatch(im_blob, im_depth_blob, depth_blob, label_blob, meta_data_blob, vertex_target_blob, pose_blob, imdb._extents, imdb._points_all, imdb._class_colors)
        data, labels, probs, vertex_pred, rois, poses, losses_values = forward_pass(sess, network, losses)
        elapsed = timeit.default_timer() - start_time
        losses_val.append(losses_values[0])
        losses_cls_val.append(losses_values[1])
        losses_vertex_val.append(losses_values[2])
        losses_pose_val.append(losses_values[3])
        print("Prefix: {}, Loss: {}, Cls: {}, Vertex: {}, Pose: {}, Time: {}".format(prefix, losses_values[0], losses_values[1], losses_values[2], losses_values[3], elapsed))
        if (i+1) % 8 == 0:
            current_iter = 796 * (4 + 1) + (i/8) + 1
            # current_iter = (i/8)
            loss_summary = sess.run(losses_ops["loss"], feed_dict={placeholders[0]: np.mean(losses_val)})
            loss_cls_summary = sess.run(losses_ops["cls"], feed_dict={placeholders[1]: np.mean(losses_cls_val)})
            loss_vertex_summary = sess.run(losses_ops["vertex"], feed_dict={placeholders[2]: np.mean(losses_vertex_val)})
            loss_pose_summary = sess.run(losses_ops["pose"], feed_dict={placeholders[3]: np.mean(losses_pose_val)})
            train_writer.add_summary(loss_summary, current_iter)
            train_writer.add_summary(loss_cls_summary, current_iter)
            train_writer.add_summary(loss_vertex_summary, current_iter)
            train_writer.add_summary(loss_pose_summary, current_iter)

        if cfg.TEST.VISUALIZE:
            vertmap = _extract_vertmap(labels, vertex_pred, imdb._extents, imdb.num_classes)
            im_label = imdb.labels_to_image(data, labels)
            poses_icp = []
            vis_segmentations_vertmaps_detection(data, depth_cv, im_label, imdb._class_colors, vertmap, labels, rois, poses, poses_icp, meta_data['intrinsic_matrix'],
                                                 imdb.num_classes, imdb._classes, imdb._points_all)


if __name__ == '__main__':
    main()
