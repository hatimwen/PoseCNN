# These two lines need to be activated when you want to record a video
# import matplotlib
# matplotlib.use('Agg')

from ros.test import _extract_vertmap, vis_segmentations_vertmaps_detection
import timeit
import tensorflow as tf
import numpy as np
from datasets.factory import get_imdb
from generate_dataset.common import get_filename_prefix, get_intrinsic_matrix
import cv2
import Image
import os
from fcn.config import cfg, cfg_from_file, get_output_dir
import pprint
import random
import yaml
from utils.blob import pad_im
from gt_synthesize_layer.minibatch import _generate_vertex_targets, _vis_minibatch
from transforms3d.quaternions import mat2quat, quat2mat
from generate_dataset.dope_to_posecnn import construct_posecnn_meta_data, get_dope_objects
import matplotlib.pyplot as plt
from ros.test import get_image_blob
from generate_dataset.dope_to_posecnn import linear_instance_segmentation_mask_image
from fcn.test import plot_data
import io
from tools.common import smooth_l1_loss_vertex, combine_poses
import scipy.io
from pyquaternion import Quaternion
import math
import itertools


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
        instance = "-object.png"
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

    return depth_blob, label_blob, meta_data_blob, vertex_target_blob, vertex_weight_blob, pose_blob, gt_boxes, mask


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
    point_blob = points.copy()
    for i in range(2):
        point_blob[i, :, :] = 40 * point_blob[i, :, :]
    feed_dict = {net.data: im_blob, net.gt_label_2d: label_blob, net.keep_prob: keep_prob, net.is_train: is_train, net.vertex_targets: vertex_target_blob,
                 net.vertex_weights: vertex_weight_blob, net.meta_data: meta_data_blob, net.extents: extents, net.points: point_blob, net.symmetry: symmetry, net.poses: pose_blob}

    sess.run(net.enqueue_op, feed_dict=feed_dict)


def forward_pass(sess, net, losses=None):
    if losses:
        loss = losses["loss"]
        loss_cls = losses["cls"]
        loss_vertex = losses["vertex"]
        loss_pose = losses["pose"]

        data, labels_2d, probs, vertex_pred, rois, poses_init, poses_pred, poses_target, loss_value, loss_cls_value, loss_vertex_value, loss_pose_value = \
            sess.run([net.get_output("data"), net.get_output('label_2d'), net.get_output('prob_normalized'), net.get_output('vertex_pred'), net.get_output('rois'),
                      net.get_output('poses_init'), net.get_output('poses_tanh'), net.get_output('poses_target'), loss, loss_cls, loss_vertex, loss_pose])
        losses_values = [loss_value[0], loss_cls_value, loss_vertex_value, sum(loss_pose_value)/40.0]
    else:
        data, labels_2d, probs, vertex_pred, rois, poses_init, poses_pred = \
            sess.run([net.get_output("data"), net.get_output('label_2d'), net.get_output('prob_normalized'), net.get_output('vertex_pred'), net.get_output('rois'),
                      net.get_output('poses_init'), net.get_output('poses_tanh')])
        poses_target = None
        losses_values = []
        loss_pose_value = None

    return combine_poses(data, rois, poses_init, poses_pred, poses_target, probs, vertex_pred, labels_2d, losses_values, loss_pose_value)


def get_instance_id(gt_poses, pose_target):
    gt_poses_quats = []
    for i in range(gt_poses.shape[2]):
        gt_orientation = Quaternion(mat2quat(gt_poses[:, :3, i]))
        gt_poses_quats.append(gt_orientation)
        if np.allclose(gt_orientation.elements, pose_target.elements, 1e-5):
            return i, gt_orientation
    print(gt_poses_quats, pose_target)


def get_errors(rois, gt_poses, poses, poses_target, slosses, batch_index=0):
    # should_print = True
    should_print = False
    instance_id_blacklist = []
    positional_errors = []
    positional_errors_vec = []
    positional_errors_depth = []
    sloss_errors = []
    should_visualize = False
    rois_pruned = []
    poses_pruned = []
    for i in xrange(rois.shape[0]):
        if rois[i][0] != batch_index:
            continue
        cx = (rois[i, 2] + rois[i, 4]) / 2
        cy = (rois[i, 3] + rois[i, 5]) / 2
        # if 635 < cx or cx < 5 or 475 < cy or cy < 5:
        #     continue
        if not np.any(poses_target[i, 4:]):
            continue
        pose_target = Quaternion(poses_target[i, 4:])
        instance_id, gt_orientation = get_instance_id(gt_poses, pose_target)
        if instance_id in instance_id_blacklist:
            continue
        else:
            instance_id_blacklist.append(instance_id)

        poses_pruned.append(poses[i])
        rois_pruned.append(rois[i])

        gt_position = gt_poses[:, 3, instance_id]

        pred_position = poses[i, 4:7]
        pred_orientation = Quaternion(poses[i, :4])
        pred_orientation = pred_orientation.normalised

        try:
            # because the points are scaled for numerical reasons by 40
            sloss = slosses[i]/40
        except IndexError:
            print("Index error")
            print(slosses)
            print("gt")
            print(gt_poses.shape)
            print("poses_target")
            print(poses_target)

        # see: https://math.stackexchange.com/questions/90081/quaternion-distance
        angle_error = math.degrees(math.acos(2*math.pow(np.inner(gt_orientation, pred_orientation), 2) - 1))
        difference_vector = gt_position - pred_position
        difference_vector_abs = np.abs(difference_vector)
        x_error = difference_vector_abs[0]
        y_error = difference_vector_abs[1]
        # if x_error > 0.2 or y_error > 0.2:
        #     should_print = True
        #     should_visualize = True
        positional_error = np.linalg.norm(difference_vector)
        positional_errors_vec.append(difference_vector)
        positional_errors.append(positional_error)
        positional_errors_depth.append(np.linalg.norm(gt_position))
        sloss_errors.append(sloss)
        if should_print:
            print("Instance_id: " + str(instance_id + 1))
            print(cx, cy)
            print(sloss)
            print(gt_position)
            print(pred_position)
            print(gt_orientation)
            print(pred_orientation)
            print("Angle: " + str(angle_error))
            print("Position error: " + str(positional_error))
    assert len(positional_errors) <= gt_poses.shape[2]
    return positional_errors, positional_errors_depth, sloss_errors, positional_errors_vec, should_visualize, np.array(rois_pruned), np.array(poses_pruned)


def prune_duplicate_boxes(rois, poses):
    rois_pruned = []
    poses_pruned = []
    for i in range(rois.shape[0]):
        pred_position = poses[i, 4:7]
        skip = False
        for pose_pruned in poses_pruned:
            if np.linalg.norm(pose_pruned[4:7] - pred_position) < 0.15:
                skip = True
                break
        if skip:
            continue
        poses_pruned.append(poses[i])
        rois_pruned.append(rois[i])
    return np.array(rois_pruned), np.array(poses_pruned)


def main():
    config_dict, sess, network, imdb, output_dir = setup()
    intrinsic_matrix = get_intrinsic_matrix()

    data_folder = config_dict["data_folder"]
    skip_frames = config_dict["skip_frames"]
    files_to_test_name = config_dict["files_to_test_name"]
    dope = False
    visualize = False
    record_video = False
    record_data = True
    # test_specific_image = 599
    # test_specific_image = 7428
    test_specific_image = None
    color, depth, cls, instance = get_postfixes(dope)

    losses, losses_ops, placeholders = build_loses(network)
    train_writer = tf.summary.FileWriter(output_dir + "/test", sess.graph)
    losses_val = []
    losses_cls_val = []
    losses_vertex_val = []
    losses_pose_val = []
    files_to_test = open(files_to_test_name)
    if dope:
        indexes = [file.split("/")[1].strip() for file in files_to_test]
    else:
        indexes = [line.strip() for line in files_to_test]
    positional_errors_all = []
    positional_errors_vec_all = []
    positional_errors_depth_all = []
    rotational_errors_all = []
    total_num_boxes = 0
    total_time = 0
    if record_video:
        out = cv2.VideoWriter('test_sim.avi', cv2.VideoWriter_fourcc(*'XVID'), 20.0, (1920, 1440))
    average_losses_over_n = len(indexes)
    for i in range(0, len(indexes), skip_frames):
        if i % average_losses_over_n == 0:
            losses_val = []
            losses_cls_val = []
            losses_vertex_val = []
            losses_pose_val = []

        if test_specific_image:
            prefix = get_filename_prefix(test_specific_image)
        else:
            prefix = indexes[i]
        src_path_prefix = os.path.join(data_folder, prefix)
        objects, meta_data = get_meta_data(dope, src_path_prefix, intrinsic_matrix)
        total_num_boxes += meta_data["poses"].shape[2]

        im, depth_cv = read_input_data(src_path_prefix, color, depth)
        # compute image blob
        im_blob, im_depth_blob, im_normal_blob, im_scale_factors, height, width = get_image_blob(im, depth_cv, meta_data, cfg)
        depth_blob, label_blob, meta_data_blob, vertex_target_blob, vertex_weight_blob, pose_blob, gt_boxes, instance_mask = \
            read_label_data(src_path_prefix, meta_data, 2, [1], imdb._extents, height, width, depth, cls, instance, objects)
        # run network
        start_time = timeit.default_timer()
        queue_up_data(sess, im_blob, network, label_blob, vertex_target_blob, vertex_weight_blob, meta_data_blob, imdb._extents, imdb._points_all, imdb._symmetry, pose_blob)
        data, labels, probs, vertex_pred, rois, poses, poses_target, losses_values, loss_pose_values = forward_pass(sess, network, losses)

        elapsed = timeit.default_timer() - start_time
        total_time += elapsed
        losses_val.append(losses_values[0])
        losses_cls_val.append(losses_values[1])
        losses_vertex_val.append(losses_values[2])
        losses_pose_val.append(losses_values[3])
        print("Prefix: {}, Loss: {}, Cls: {}, Vertex: {}, Pose: {}, Time: {}".format(prefix, losses_values[0], losses_values[1], losses_values[2], losses_values[3], elapsed))
        if (i+1) % average_losses_over_n == 0:
            current_iter = 796 * (4 + 1) + (i/8) + 1
            # current_iter = (i/8)
            loss_summary = sess.run(losses_ops["loss"], feed_dict={placeholders[0]: np.mean(losses_val)})
            loss_cls_summary = sess.run(losses_ops["cls"], feed_dict={placeholders[1]: np.mean(losses_cls_val)})
            loss_vertex_summary = sess.run(losses_ops["vertex"], feed_dict={placeholders[2]: np.mean(losses_vertex_val)})
            loss_pose_summary = sess.run(losses_ops["pose"], feed_dict={placeholders[3]: np.mean(losses_pose_val)})
            losses_str = "Mean losses(total, cls, vertex, pose): {}, {}, {}, {}".format(np.mean(losses_val), np.mean(losses_cls_val), np.mean(losses_vertex_val), np.mean(losses_pose_val))
            print(losses_str)
            train_writer.add_summary(loss_summary, current_iter)
            train_writer.add_summary(loss_cls_summary, current_iter)
            train_writer.add_summary(loss_vertex_summary, current_iter)
            train_writer.add_summary(loss_pose_summary, current_iter)

        positional_errors, positional_errors_depth, sloss, positional_errors_vec, _, rois_pruned, poses_pruned = get_errors(rois, meta_data['poses'], poses, poses_target, loss_pose_values)
        rois_pruned, poses_pruned = prune_duplicate_boxes(rois, poses)
        positional_errors_all.append(positional_errors)
        positional_errors_depth_all.append(positional_errors_depth)
        rotational_errors_all.append(sloss)
        positional_errors_vec_all.append(positional_errors_vec)

        if visualize:
            _vis_minibatch(im_blob, im_depth_blob, depth_blob, label_blob, meta_data_blob, vertex_target_blob, pose_blob, imdb._extents, imdb._points_all, imdb._class_colors)
            vertmap = _extract_vertmap(labels, vertex_pred, imdb._extents, imdb.num_classes)
            im_label = imdb.labels_to_image(data, labels)
            poses_icp = []
            vis_segmentations_vertmaps_detection(data, depth_cv, im_label, imdb._class_colors, vertmap, labels, rois_pruned, poses_pruned, poses_icp, meta_data['intrinsic_matrix'],
                                                 imdb.num_classes, imdb._classes, imdb._points_all, None)
            if record_video:
                buf = io.BytesIO()
                plt.savefig(buf, format='png', dpi=300)
                buf.seek(0)
                plt_img = Image.open(buf)
                plt_img = cv2.cvtColor(np.array(plt_img), cv2.COLOR_RGBA2BGR)
                out.write(np.array(plt_img))
                buf.close()
                plt.close("all")
            else:
                plt.show()

    if record_video:
        out.release()
    if record_data:
        print(total_num_boxes)
        print(total_time)
        np.save("histo_position", list(itertools.chain(*positional_errors_all)))
        np.save("histo_vec", list(itertools.chain(*positional_errors_vec_all)))
        np.save("histo_distance", list(itertools.chain(*positional_errors_depth_all)))
        np.save("histo_sloss", list(itertools.chain(*rotational_errors_all)))
        with open("meta_data.txt", "w") as f:
            f.write("Total boxes: " + str(total_num_boxes) + "\n")
            f.write("Total time: " + str(total_time) + "\n")
            losses_str = "Mean losses(total, cls, vertex, pose): {}, {}, {}, {}".format(np.mean(losses_val), np.mean(losses_cls_val), np.mean(losses_vertex_val),
                                                                                        np.mean(losses_pose_val))
            f.write(losses_str + "\n")


if __name__ == '__main__':
    main()
