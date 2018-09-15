# -*- coding: utf-8 -*-
# --------------------------------------------------------
# Faster R-CNN
# Written by KimSeongJung
# --------------------------------------------------------
import sys
import numpy as np
import numpy.random as npr
import tensorflow as tf
import bbox_overlaps
import bbox_transform
import generate_anchors



def anchor_target(topconv_h , topconv_w , gt_boxes, im_dims, _feat_stride, anchor_scales):
    with tf.variable_scope('anchor_target'):
        labels, bbox_targets, bbox_inside_weights, bbox_outside_weights = tf.py_func(anchor_target_py,
                                                    [topconv_h, topconv_w, gt_boxes,im_dims, _feat_stride, anchor_scales],
                                                    [tf.float32, tf.float32, tf.float32,tf.float32])


        labels = tf.convert_to_tensor(tf.cast(labels, tf.int32), name='labels')
        bbox_targets = tf.convert_to_tensor(bbox_targets, name='bbox_targets')
        bbox_inside_weights = tf.convert_to_tensor(bbox_inside_weights, name='bbox_inside_weights')
        bbox_outside_weights = tf.convert_to_tensor(bbox_outside_weights, name='bbox_outside_weights')
        return  labels, bbox_targets, bbox_inside_weights, bbox_outside_weights


def anchor_target_py(topconv_h , topconv_w , gt_boxes, im_dims, _feat_stride, anchor_scales):
    """
    1.anchor 생성
    2.
    :return:
    """
    im_dims = im_dims[0]
    _allowed_border = 0

    # Get Anchors
    _anchors = generate_anchors.generate_anchors(scales=np.array(anchor_scales))
    _num_anchors = _anchors.shape[0]

    # Get Shifts
    shifts = _create_grid(topconv_h , topconv_w , _feat_stride)

    # Mapping Anchors to Shifts
    all_anchors =_mapping_anchors( _anchors , shifts )
    all_anchors = all_anchors.reshape((-1, 4))
    total_anchors = len(all_anchors)

    ## Delete out of boundary
    inds_inside = _out_of_boundary(all_anchors , _allowed_border , im_dims)
    anchors = all_anchors[inds_inside]

    # Get Overlaps
    overlaps = bbox_overlaps.bbox_overlaps(np.ascontiguousarray(anchors, dtype=np.float),
                                           np.ascontiguousarray(gt_boxes, dtype=np.float))
    # Get max_overlaps
    max_overlaps, argmax_overlaps = _get_max_overlaps(overlaps , inds_inside)
    # Get argmax overlaps
    gt_max_overlaps, gt_argmax_overlaps_inds = _get_gt_overlaps(overlaps)

    # Settings
    RPN_NEGATIVE_OVERLAP = 0.3
    RPN_POSITIVE_OVERLAP = 0.7
    RPN_FG_FRACTION = 0.5
    RPN_BATCHSIZE = 60
    RPN_BBOX_INSIDE_WEIGHTS = (1.0, 1.0, 1.0, 1.0)
    RPN_POSITIVE_WEIGHT = -1.0

    # Set Labels
    labels = np.empty((len(inds_inside),), dtype=np.float32)
    labels = _set_fg_bg(labels, max_overlaps, gt_argmax_overlaps_inds, RPN_NEGATIVE_OVERLAP, RPN_POSITIVE_OVERLAP)
    labels = _balance_fg_bg(labels, RPN_BATCHSIZE , RPN_FG_FRACTION)

    # Get targets from anchos
    bbox_targets = _compute_targets(anchors, gt_boxes[argmax_overlaps, :])

    # Get inside Weights
    bbox_inside_weights = np.zeros((len(inds_inside), 4), dtype=np.float32)
    bbox_inside_weights = _set_inside_weights(bbox_inside_weights ,labels , RPN_BBOX_INSIDE_WEIGHTS)

    # Get outside Weights
    bbox_outside_weights = np.zeros((len(inds_inside), 4), dtype=np.float32)
    bbox_outside_weights = _set_outside_weights(bbox_outside_weights , labels, RPN_POSITIVE_WEIGHT)

    # unmap
    labels = _unmap(labels, total_anchors, inds_inside, fill=-1)
    bbox_targets = _unmap(bbox_targets, total_anchors, inds_inside, fill=0)
    bbox_inside_weights = _unmap(bbox_inside_weights, total_anchors, inds_inside, fill=0)
    bbox_outside_weights = _unmap(bbox_outside_weights, total_anchors, inds_inside, fill=0)

    # label을 변환한다
    labels.reshape((1, topconv_h, topconv_w, _num_anchors)).transpose(0, 3, 1, 2) # A = 9 , (1,h,w,9) ==> (1,9,h,w)
    labels = labels.reshape((1, 1, _num_anchors * topconv_h, topconv_w)) # 왜 변하지 ?

    return labels , bbox_targets , bbox_inside_weights, bbox_outside_weights


def _mapping_anchors(anchors , shifts):
    all_anchors=np.array([])
    for i in range(len(anchors)):
        if i ==0 :
            all_anchors=np.add(shifts , anchors[i])
        else:
            all_anchors = np.concatenate((all_anchors, np.add(shifts, anchors[i])), axis=0)
    return all_anchors


def _create_grid(height, width , feat_stride):
    shift_x = np.arange(0, width) * feat_stride
    shift_y = np.arange(0, height) * feat_stride
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)
    shifts = np.vstack((shift_x.ravel(), shift_y.ravel(),
                        shift_x.ravel(), shift_y.ravel())).transpose() # 4,88 을 88,4 로 바꾼다
    return shifts


def _out_of_boundary(all_anchors , _allowed_border , im_dims):
    inds_inside = np.where(
        (all_anchors[:, 0] >= -_allowed_border) &
        (all_anchors[:, 1] >= -_allowed_border) &
        (all_anchors[:, 2] < im_dims[1] + _allowed_border) &  # <-- width
        (all_anchors[:, 3] < im_dims[0] + _allowed_border))[0] # <-- height
    return inds_inside


def _get_max_overlaps(overlaps , inds_inside):
    argmax_overlaps = overlaps.argmax(axis=1)
    max_overlaps = overlaps[np.arange(len(inds_inside)), argmax_overlaps]  # inds_inside 갯수 만큼 overlaps에서 가장 높은 overlays
    return max_overlaps , argmax_overlaps

def _get_gt_overlaps(overlaps):
    # 모든 overlaps 중에서 가장 많이 겹치는 것을 가져온다
    gt_argmax_overlaps = overlaps.argmax(axis=0)

    # 가장 많이 겹치는 overlab 의 overlap 비율을 가져온다, [ 0.63126253  0.76097561]
    gt_max_overlaps = overlaps[gt_argmax_overlaps, np.arange(overlaps.shape[1])]  # *

    # 가장 많이 겹치는 overlab 의 arg 을 가져온다
    gt_argmax_overlaps_inds = np.where(overlaps == gt_max_overlaps)[0]

    return gt_max_overlaps , gt_argmax_overlaps_inds


def _balance_fg_bg(labels , batch_size, fg_fraction):    # Training Set 에서  foreground 와 background 의 비율을 맞춘다
    num_fg = int(fg_fraction* batch_size)  # RPN_FG_FRACTION = 0.5
    fg_inds = np.where(labels == 1)[0]
    if len(fg_inds) > num_fg:
        disable_inds = npr.choice(
            fg_inds, size=(len(fg_inds) - num_fg), replace=False)  # replace = False --> 겹치지 않게 한다
        labels[disable_inds] = -1
    else: # same or less
        num_fg = len(fg_inds)
    # subsample negative labels if we have too many

    # 1:1 = fg and bg
    num_bg = num_fg
    bg_inds = np.where(labels == 0)[0]
    if len(bg_inds) > num_bg:
        disable_inds = npr.choice(bg_inds, size=len((bg_inds)) - num_bg , replace=False)
        labels[disable_inds] = -1
    else: # num_bg > num_fg
        pass;
    assert np.sum([labels == 0]) == np.sum([labels == 1]), '{} {} {}'.format(np.sum([labels == 0]),
                                                                             np.sum([labels == 1]),
                                                                             np.sum([labels == -1]))
    return labels


def _set_fg_bg(labels , max_overlaps , gt_argmax_overlaps , neg_overlap , pos_overlap):
    labels.fill(-1)
    labels[max_overlaps < neg_overlap] = 0
    # fg label: for each gt, anchor with highest overlap
    labels[gt_argmax_overlaps] = 1 # 가장 높은 anchor의 라벨은 1로 준다
    # fg label: above threshold IOU
    labels[max_overlaps >= pos_overlap] = 1
    return labels



def _set_inside_weights(bbox_inside_weights , labels , inside_weight):
    """

    :param bbox_inside_weights:
    :param labels:
    :param inside_weights: (1.0, 1.0, 1.0, 1.0)
    :return:
    """
    bbox_inside_weights[labels == 1, :] = np.array(inside_weight)
    return bbox_inside_weights

def _set_outside_weights(bbox_outside_weights , labels , positive_weight):
    """

    :param bbox_outside_weights:
    :param labels:
    :param positive_weights:
    :return:
    """
    if positive_weight < 0:  # TRAIN.RPN_POSITIVE_WEIGHT = -1
        # uniform weighting of examples (given non-uniform sampling)
        num_examples = np.sum(labels >= 0)  # get positive label
        positive_weights = np.ones((1, 4)) * 1.0 / num_examples
        negative_weights = np.ones((1, 4)) * 1.0 / num_examples
    else:
        assert ((positive_weight> 0) & (positive_weight< 1))
        positive_weights = (positive_weight / np.sum(labels == 1))
        negative_weights = ((1.0 - positive_weight) / np.sum(labels == 0))

    bbox_outside_weights[labels == 1, :] = positive_weights
    bbox_outside_weights[labels == 0, :] = negative_weights
    return bbox_outside_weights

def _unmap(data, count, inds, fill=0):
    """ Unmap a subset of item (data) back to the original set of items (of size count) """
    if len(data.shape) == 1:
        ret = np.empty((count,), dtype=np.float32)
        ret.fill(fill)
        ret[inds] = data
    else:
        ret = np.empty((count,) + data.shape[1:], dtype=np.float32)
        ret.fill(fill)
        ret[inds, :] = data
    return ret


def _compute_targets(ex_rois, gt_rois):
    """Compute bounding-box regression targets for an image."""

    assert ex_rois.shape[0] == gt_rois.shape[0]
    assert ex_rois.shape[1] == 4 # anchor
    assert gt_rois.shape[1] == 5 # gt_bbox

    return bbox_transform.bbox_transform(ex_rois, gt_rois[:, :4]).astype(np.float32, copy=False)