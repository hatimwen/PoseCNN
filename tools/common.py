import tensorflow as tf
import numpy as np


def smooth_l1_loss_vertex(vertex_pred, vertex_targets, vertex_weights, sigma=1.0):
    sigma_2 = sigma ** 2
    vertex_diff = vertex_pred - vertex_targets
    diff = tf.multiply(vertex_weights, vertex_diff)
    abs_diff = tf.abs(diff)
    smoothL1_sign = tf.stop_gradient(tf.to_float(tf.less(abs_diff, 1. / sigma_2)))
    in_loss = tf.pow(diff, 2) * (sigma_2 / 2.) * smoothL1_sign \
              + (abs_diff - (0.5 / sigma_2)) * (1. - smoothL1_sign)
    loss = tf.div(tf.reduce_sum(in_loss), tf.reduce_sum(vertex_weights) + 1e-10)
    return loss


def combine_poses(data, rois, poses_init, poses_pred, probs, vertex_pred, labels_2d):
    # combine poses
    num = rois.shape[0]
    poses = poses_init
    for i in xrange(num):
        class_id = int(rois[i, 1])
        if class_id >= 0:
            poses[i, :4] = poses_pred[i, 4 * class_id:4 * class_id + 4]
    vertex_pred = vertex_pred[0, :, :, :]
    return data, labels_2d[0, :, :].astype(np.int32), probs[0, :, :, :], vertex_pred, rois, poses