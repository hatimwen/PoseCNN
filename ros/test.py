import rospy
import message_filters
import cv2
import numpy as np
from fcn.config import cfg
from fcn.test import vis_segmentations_vertmaps_detection, _extract_vertmap
from utils.blob import im_list_to_blob, pad_im, unpad_im, add_noise
from normals import gpu_normals
from std_msgs.msg import String
from sensor_msgs.msg import Image
from transforms3d.quaternions import quat2mat
import matplotlib.pyplot as plt
import timeit
import tensorflow as tf
from tools.common import smooth_l1_loss_vertex, combine_poses


def test_ros(sess, network, imdb, meta_data, cfg, rgb, depth, cv_bridge, count):
    if depth is not None:
        if depth.encoding == '32FC1':
            depth_32 = cv_bridge.imgmsg_to_cv2(depth) * 1000
            depth_cv = np.array(depth_32, dtype=np.uint16)
        elif depth.encoding == '16UC1':
            depth_cv = cv_bridge.imgmsg_to_cv2(depth)
        else:
            rospy.logerr_throttle(1, 'Unsupported depth type. Expected 16UC1 or 32FC1, got {}'.format(depth.encoding))
            return
    else:
        depth_cv = None

    # image
    im = cv_bridge.imgmsg_to_cv2(rgb, 'bgr8')

    # write images
    filename = 'images/%06d-color.png' % count
    cv2.imwrite(filename, im)

    if depth is not None:
        filename = 'images/%06d-depth.png' % count
        cv2.imwrite(filename, depth_cv)
        print filename

    # run network
    start_time = timeit.default_timer()
    labels, probs, vertex_pred, rois, poses, poses_gt = im_segment_single_frame(sess, network, im, depth_cv, meta_data, \
            imdb._extents, imdb._points_all, imdb._symmetry, imdb.num_classes, cfg)
    elapsed = timeit.default_timer() - start_time
    print("ELAPSED TIME:")
    print(elapsed)
    poses_icp = []

    im_label = imdb.labels_to_image(im, labels)

    if cfg.TEST.VISUALIZE:
        vertmap = _extract_vertmap(labels, vertex_pred, imdb._extents, imdb.num_classes)
        # vis_segmentations_vertmaps(im, depth_cv, im_label, imdb._class_colors, \
        #             vertmap, labels, rois, poses, poses_icp, meta_data['intrinsic_matrix'], \
        #             imdb.num_classes, imdb._points_all, cfg)
        vis_segmentations_vertmaps_detection(im, depth_cv, im_label, imdb._class_colors, \
                                   vertmap, labels, rois, poses, poses_icp, meta_data['intrinsic_matrix'], \
                                   imdb.num_classes, imdb._classes, imdb._points_all, poses_gt)


def get_image_blob(im, im_depth, meta_data, cfg):
    """Converts an image into a network input.

    Arguments:
        im (ndarray): a color image in BGR order

    Returns:
        blob (ndarray): a data blob holding an image pyramid
        im_scale_factors (list): list of image scales (relative to im) used
            in the image pyramid
    """

    # RGB
    im_orig = im.astype(np.float32, copy=True)
    im_orig -= cfg.PIXEL_MEANS
    processed_ims = []
    im_scale_factors = []
    assert len(cfg.TEST.SCALES_BASE) == 1
    im_scale = cfg.TEST.SCALES_BASE[0]

    im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR)
    im_scale_factors.append(im_scale)
    processed_ims.append(im)
    # Create a blob to hold the input images
    blob = im_list_to_blob(processed_ims, 3)
    height = processed_ims[0].shape[0]
    width = processed_ims[0].shape[1]

    # depth
    if im_depth is not None:
        im_orig = im_depth.astype(np.float32, copy=True)
        im_orig = im_orig / im_orig.max() * 255
        im_orig = np.tile(im_orig[:,:,np.newaxis], (1,1,3))
        im_orig -= cfg.PIXEL_MEANS

        processed_ims_depth = []
        im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR)
        processed_ims_depth.append(im)
        blob_depth = im_list_to_blob(processed_ims_depth, 3)
    else:
        blob_depth = None

    if cfg.INPUT == 'NORMAL':
        # meta data
        K = meta_data['intrinsic_matrix'].astype(np.float32, copy=True)
        fx = K[0, 0]
        fy = K[1, 1]
        cx = K[0, 2]
        cy = K[1, 2]

        # normals
        depth = im_depth.astype(np.float32, copy=True) / float(meta_data['factor_depth'])
        nmap = gpu_normals.gpu_normals(depth, fx, fy, cx, cy, 20.0, cfg.GPU_ID)
        im_normal = 127.5 * nmap + 127.5
        im_normal = im_normal.astype(np.uint8)
        im_normal = im_normal[:, :, (2, 1, 0)]
        im_normal = cv2.bilateralFilter(im_normal, 9, 75, 75)

        processed_ims_normal = []
        im_orig = im_normal.astype(np.float32, copy=True)
        im_orig -= cfg.PIXEL_MEANS
        im_normal = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR)
        processed_ims_normal.append(im_normal)
        # Create a blob to hold the input images
        blob_normal = im_list_to_blob(processed_ims_normal, 3)
    else:
        blob_normal = []
        
    return blob, blob_depth, blob_normal, np.array(im_scale_factors), height, width


def get_data(sess, net, losses, output_dir, current_iter):
    loss = losses["loss"]
    loss_cls = losses["cls"]
    loss_vertex = losses["vertex"]
    loss_pose = losses["pose"]

    loss_op = tf.summary.scalar('loss', tf.squeeze(loss))
    loss_cls_op = tf.summary.scalar('loss_cls', tf.squeeze(loss_cls))
    loss_vertex_op = tf.summary.scalar('loss_vertex', tf.squeeze(loss_vertex))
    loss_pose_op = tf.summary.scalar('loss_pose', tf.squeeze(loss_pose))

    data, labels_2d, probs, vertex_pred, rois, poses_init, pool_score, pool5, pool4, poses_pred, poses_pred2, loss_summary, loss_cls_summary, \
    loss_vertex_summary, loss_pose_summary, loss_value, loss_cls_value, loss_vertex_value, loss_pose_value = \
        sess.run([net.get_output("data"), net.get_output('label_2d'), net.get_output('prob_normalized'), net.get_output('vertex_pred'), \
                  net.get_output('rois'), net.get_output('poses_init'), net.get_output("pool_score"), net.get_output("pool5"), net.get_output("pool4"),
                  net.get_output('poses_tanh'), net.get_output("poses_pred"), loss_op, loss_cls_op, loss_vertex_op, loss_pose_op, loss, loss_cls, loss_vertex, loss_pose])
    train_writer = tf.summary.FileWriter(output_dir + "/test", sess.graph)
    print("Loss: ", loss_value[0])
    print("Cls: ", loss_cls_value)
    print("Vertex: ", loss_vertex_value)
    print("Pose: ", loss_pose_value[0])

    train_writer.add_summary(loss_summary, current_iter)
    train_writer.add_summary(loss_cls_summary, current_iter)
    train_writer.add_summary(loss_vertex_summary, current_iter)
    train_writer.add_summary(loss_pose_summary, current_iter)

    return combine_poses(data, rois, poses_init, poses_pred, probs, vertex_pred, labels_2d)


def im_segment_single_frame(sess, net, im_blob, im_depth_blob, im_normal_blob, meta_data, extents, points, symmetry, num_classes, cfg, output_dir, i, depth_blob, label_blob, meta_data_blob, vertex_target_blob, vertex_weight_blob, pose_blob, gt_boxes):
    """segment image
    """

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
        "cls": loss_cls,
        "vertex": loss_vertex,
        "pose": loss_pose,
        "loss": loss
    }

    # forward pass
    if cfg.INPUT == 'RGBD':
        data_blob = im_blob
        data_p_blob = im_depth_blob
    elif cfg.INPUT == 'COLOR':
        data_blob = im_blob
    elif cfg.INPUT == 'DEPTH':
        data_blob = im_depth_blob
    elif cfg.INPUT == 'NORMAL':
        data_blob = im_normal_blob

    keep_prob = 1.0
    is_train = False

    if cfg.INPUT == 'RGBD':
        if cfg.TEST.VERTEX_REG_2D or cfg.TEST.VERTEX_REG_3D:
            feed_dict = {net.data: data_blob, net.data_p: data_p_blob, net.gt_label_2d: label_blob, net.keep_prob: keep_prob, \
                         net.vertex_targets: vertex_target_blob, net.vertex_weights: vertex_weight_blob, \
                         net.meta_data: meta_data_blob, net.extents: extents, net.points: points, net.poses: pose_blob}
        else:
            feed_dict = {net.data: data_blob, net.data_p: data_p_blob, net.gt_label_2d: label_blob, net.keep_prob: keep_prob}
    else:
        if cfg.TEST.VERTEX_REG_2D or cfg.TEST.VERTEX_REG_3D:
            feed_dict = {net.data: data_blob, net.gt_label_2d: label_blob, net.keep_prob: keep_prob, net.is_train: is_train, \
                         net.vertex_targets: vertex_target_blob, net.vertex_weights: vertex_weight_blob, \
                         net.meta_data: meta_data_blob, net.extents: extents, net.points: points, net.symmetry: symmetry, net.poses: pose_blob}
        else:
            feed_dict = {net.data: data_blob, net.gt_label_2d: label_blob, net.keep_prob: keep_prob}

    sess.run(net.enqueue_op, feed_dict=feed_dict)

    if cfg.TEST.VERTEX_REG_2D:
        if cfg.TEST.POSE_REG:
            return get_data(sess, net, losses, output_dir, i)
        else:
            labels_2d, probs, vertex_pred, rois, poses = \
                sess.run([net.get_output('label_2d'), net.get_output('prob_normalized'), net.get_output('vertex_pred'), net.get_output('rois'), net.get_output('poses_init')])
            print rois
            print rois.shape
            # non-maximum suppression
            # keep = nms(rois[:, 2:], 0.5)
            # rois = rois[keep, :]
            # poses = poses[keep, :]

            #labels_2d, probs = sess.run([net.get_output('label_2d'), net.get_output('prob_normalized')])
            #vertex_pred = []
            #rois = []
            #poses = []
            vertex_pred = vertex_pred[0, :, :, :]
    else:
        labels_2d, probs = sess.run([net.get_output('label_2d'), net.get_output('prob_normalized')])
        vertex_pred = []
        rois = []
        poses = []

    return labels_2d[0,:,:].astype(np.int32), probs[0,:,:,:], vertex_pred, rois, poses



def vis_segmentations_vertmaps(im, im_depth, im_labels, colors, center_map, 
  labels, rois, poses, poses_new, intrinsic_matrix, num_classes, points, cfg):
    """Visual debugging of detections."""
    fig = plt.figure()

    # show image
    ax = fig.add_subplot(3, 4, 1)
    im = im[:, :, (2, 1, 0)]
    plt.imshow(im)
    ax.set_title('input image')

    # show class label
    ax = fig.add_subplot(3, 4, 9)
    plt.imshow(im_labels)
    ax.set_title('class labels')      

    if cfg.TEST.VERTEX_REG_2D:
        # show centers
        for i in xrange(rois.shape[0]):
            if rois[i, 1] == 0:
                continue
            cx = (rois[i, 2] + rois[i, 4]) / 2
            cy = (rois[i, 3] + rois[i, 5]) / 2
            w = rois[i, 4] - rois[i, 2]
            h = rois[i, 5] - rois[i, 3]
            if not np.isinf(cx) and not np.isinf(cy):
                plt.plot(cx, cy, 'yo')

                # show boxes
                plt.gca().add_patch(
                    plt.Rectangle((cx-w/2, cy-h/2), w, h, fill=False,
                                   edgecolor='g', linewidth=3))
        
    # show vertex map
    ax = fig.add_subplot(3, 4, 10)
    plt.imshow(center_map[:,:,0])
    ax.set_title('centers x')

    ax = fig.add_subplot(3, 4, 11)
    plt.imshow(center_map[:,:,1])
    ax.set_title('centers y')
    
    ax = fig.add_subplot(3, 4, 12)
    plt.imshow(center_map[:,:,2])
    ax.set_title('centers z: {:6f}'.format(poses[0, 6]))

    # show projection of the poses
    if cfg.TEST.POSE_REG:

        ax = fig.add_subplot(3, 4, 3, aspect='equal')
        plt.imshow(im)
        ax.invert_yaxis()
        for i in xrange(rois.shape[0]):
            cls = int(rois[i, 1])
            if cls > 0:
                # extract 3D points
                x3d = np.ones((4, points.shape[1]), dtype=np.float32)
                x3d[0, :] = points[cls,:,0]
                x3d[1, :] = points[cls,:,1]
                x3d[2, :] = points[cls,:,2]

                # projection
                RT = np.zeros((3, 4), dtype=np.float32)
                RT[:3, :3] = quat2mat(poses[i, :4])
                RT[:, 3] = poses[i, 4:7]
                x2d = np.matmul(intrinsic_matrix, np.matmul(RT, x3d))
                x2d[0, :] = np.divide(x2d[0, :], x2d[2, :])
                x2d[1, :] = np.divide(x2d[1, :], x2d[2, :])
                plt.plot(x2d[0, :], x2d[1, :], '.', color=np.divide(colors[cls], 255.0), alpha=0.05)
                # plt.scatter(x2d[0, :], x2d[1, :], marker='o', color=np.divide(colors[cls], 255.0), s=10)

        ax.set_title('projection')
        ax.invert_yaxis()
        ax.set_xlim([0, im.shape[1]])
        ax.set_ylim([im.shape[0], 0])

        if cfg.TEST.POSE_REFINE:
            ax = fig.add_subplot(3, 4, 4, aspect='equal')
            plt.imshow(im)
            ax.invert_yaxis()
            for i in xrange(rois.shape[0]):
                cls = int(rois[i, 1])
                if cls > 0:
                    # extract 3D points
                    x3d = np.ones((4, points.shape[1]), dtype=np.float32)
                    x3d[0, :] = points[cls,:,0]
                    x3d[1, :] = points[cls,:,1]
                    x3d[2, :] = points[cls,:,2]

                    # projection
                    RT = np.zeros((3, 4), dtype=np.float32)
                    RT[:3, :3] = quat2mat(poses_new[i, :4])
                    RT[:, 3] = poses_new[i, 4:7]
                    x2d = np.matmul(intrinsic_matrix, np.matmul(RT, x3d))
                    x2d[0, :] = np.divide(x2d[0, :], x2d[2, :])
                    x2d[1, :] = np.divide(x2d[1, :], x2d[2, :])
                    plt.plot(x2d[0, :], x2d[1, :], '.', color=np.divide(colors[cls], 255.0), alpha=0.05)

            ax.set_title('projection refined by ICP')
            ax.invert_yaxis()
            ax.set_xlim([0, im.shape[1]])
            ax.set_ylim([im.shape[0], 0])

    plt.show()
