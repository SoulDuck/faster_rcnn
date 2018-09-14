import tensorflow as tf
import numpy as np
from generate_anchors import generate_anchors
from bbox_transform import bbox_transform_inv , clip_boxes
def rpn_softmax(rpn_cls_layer):
    """
    :param rpn_cls_layer: 1.shape : (1, h ,w ,18 )
    :return:
    """
    # 1, h ,w , 18
    shape = tf.shape(rpn_cls_layer)
    # (1, h ,w , 18) ==> (1, 18 ,h ,w)
    rpn_cls_score = tf.transpose(rpn_cls_layer, [0, 3, 1, 2])
    # (1, 18 ,h ,w) ==> (1, 2 ,h*9 ,w)
    rpn_cls_score = tf.reshape(rpn_cls_score, [shape[0], 2, shape[3] // 2 * shape[1],shape[2]])
    # (1, 2 ,h*9 ,w) ==> (1, h*9 ,w ,2)
    rpn_cls_score = tf.transpose(rpn_cls_score,[0, 2, 3, 1]) # shape=(?, h*9, w, 2)
    # (1, h*9 ,w ,2) ==> (h*9*w ,2)
    rpn_cls_score = tf.reshape(rpn_cls_score, [-1, 2])# shape=(h*9*w, 2)
    # softmax
    rpn_cls_prob = tf.nn.softmax(rpn_cls_score) # # shape=(h*9*w, 2)

    # back to the original shape
    # (h*9*w ,2) ==> (1, 9*h , w, 2)
    rpn_cls_prob = tf.reshape(rpn_cls_prob, [1,( shape[3] // 2 )*shape[1], shape[2],  2])
    # (1, 9*h , w, 2) ==> (1, 2 , h*9, w)
    rpn_cls_prob = tf.transpose(rpn_cls_prob, [0,3,1,2])
    # (1, 2 , h*9, w) ==> (1, 2*9 , h, w)
    rpn_cls_prob = tf.reshape(rpn_cls_prob, [1, 2*(shape[3] // 2), shape[1],  shape[2]]) #(1,18 , h, w)
    # (1, 2*9 , h, w) ==> (1, h, w , 18 )
    rpn_cls_prob = tf.transpose(rpn_cls_prob, [0,2,3,1])  # (1, h, w , 18)

    return rpn_cls_prob

def roi_proposal(rpn_cls_prob , rpn_bbox_layer, im_dims , _feat_stride , anchor_scales , is_training):
    """
    :param rpn_cls_prob:  shape :
    :param rpn_bbox_layer: shape :

    :param im_dims:
    :param _feat_stride:
    :param anchor_scales:
    :param is_training:
    :return:
    """
    print '########################################################'
    print '########## ROI Proposal Network building.... ###########'
    print '########################################################'

    blobs, scores = proposal_layer(rpn_cls_prob=rpn_cls_prob, rpn_bbox_pred=rpn_bbox_layer, im_dims=im_dims,
                                   _feat_stride=_feat_stride,
                                   anchor_scales=anchor_scales,
                                   is_training=is_training)

    return blobs , scores


def proposal_layer(rpn_cls_prob, rpn_bbox_pred, im_dims, _feat_stride, anchor_scales ,is_training):
    blobs , scores = tf.py_func( _proposal_layer_py,
                                 [rpn_cls_prob, rpn_bbox_pred, im_dims[0], _feat_stride, anchor_scales , is_training],
                                 [tf.float32 , tf.float32 ])
    blobs=tf.reshape(blobs , shape=[-1,5] , name='roi_blobs_op')
    scores=tf.reshape(scores , shape=[-1] , name='roi_scores_op')
    return blobs ,  scores



def _filter_boxes(boxes, min_size):
    """Remove all boxes with any side smaller than min_size."""
    ws = boxes[:, 2] - boxes[:, 0] + 1
    hs = boxes[:, 3] - boxes[:, 1] + 1
    keep = np.where((ws >= min_size) & (hs >= min_size))[0]
    return keep

def _proposal_layer_py(rpn_cls_prob, rpn_bbox_pred, im_dims, _feat_stride, anchor_scales , is_training):
    '''
    # input Shape
    # param : rpn_cls_prob shape : 1 , h , w , 2*9
    # param : rpn_bbox_pred shape : 1 , h , w , 4*9

    # function:
    # 1.clip predicted boxes to image
    # 2.remove predicted boxes with either height or width < threshold
    # 3.sort all (proposal, score) pairs by score from highest to lowest
    # 4.take top pre_nms_topN proposals before NMS
    # 5.apply NMS with threshold 0.7 to remaining proposals
    '''

    # generate Anchors
    _anchors = generate_anchors(scales=np.array(anchor_scales))
    _num_anchors = _anchors.shape[0]

    # settings
    if is_training:
        pre_nms_topN = 12000
        min_size = 16

    else:  # cfg_key == 'TEST':
        pre_nms_topN = 6000
        min_size = 16

    # Transform
    #  (1, h , w ,18) ==> (1, 18 , h , w)
    rpn_cls_prob = np.transpose(rpn_cls_prob, [0, 3, 1, 2])
    #  (1, h , w ,36) ==> (1, 36 , h , w)
    rpn_bbox_pred = np.transpose(rpn_bbox_pred, [0, 3, 1, 2])
    #
    assert rpn_cls_prob.shape[0] == 1, \
        'Only single item batches are supported'

    # the first set of _num_anchors channels are bg probs
    # the second set are the fg probs
    # 1. Generate proposals from bbox deltas and shifted anchors
    n, ch , height, width = rpn_cls_prob.shape

    # cls Transform
    scores = rpn_cls_prob.reshape([1,2, ch//2 *  height ,width])
    scores = scores.transpose([0,2,3,1])
    scores = scores.reshape([-1,2])
    scores = scores[:,1]
    scores = scores.reshape([-1,1])

    # bbox Transform
    shape = rpn_bbox_pred.shape # 1,4*A , H, W
    rpn_bbox_pred=rpn_bbox_pred.reshape([1, 4 , (shape[1]//4)*shape[2] , shape[3] ])
    rpn_bbox_pred=rpn_bbox_pred.transpose([0,2,3,1])
    rpn_bbox_pred = rpn_bbox_pred.reshape([-1,4])
    bbox_deltas = rpn_bbox_pred

    # Generate Anchors & Enumerate all shifts
    shift_x = np.arange(0, width) * _feat_stride
    shift_y = np.arange(0, height) * _feat_stride
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)
    shifts = np.vstack((shift_x.ravel(), shift_y.ravel(),
                        shift_x.ravel(), shift_y.ravel())).transpose()
    A = _num_anchors
    K = shifts.shape[0]
    anchors = np.array([])
    for i in range(len(_anchors)):
        if i == 0:
            anchors = np.add(shifts, _anchors[i])
        else:
            anchors = np.concatenate((anchors, np.add(shifts, _anchors[i])), axis=0)
    anchors = anchors.reshape((K * A, 4))

    ## BBOX TRANSPOSE Using Anchor
    proposals = bbox_transform_inv(anchors, bbox_deltas)
    proposals = clip_boxes(proposals, im_dims)
    keep = _filter_boxes(proposals, min_size)
    proposals = proposals[keep, :]
    scores = scores[keep]
    order = scores.ravel().argsort()[::-1]
    if pre_nms_topN > 0:
        order = order[:pre_nms_topN]
    scores = scores[order]
    proposals = proposals[order]

    batch_inds = np.zeros((proposals.shape[0], 1), dtype=np.float32)
    blob = np.hstack((batch_inds, proposals.astype(np.float32, copy=False))) # N , 5
    return blob , scores
