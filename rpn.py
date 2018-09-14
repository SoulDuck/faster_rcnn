import tensorflow as tf
import numpy as np
from cnn import convolution2d
def rpn_cls_layer(layer , n_anchors = 9 ):
    with tf.variable_scope('cls'):
        layer = convolution2d('rpn_cls_conv' ,layer, out_ch= n_anchors*2 , k=1 , act=None , s=1)
        layer = tf.identity(layer, name='cls_output')
        print '** cls layer shape : {}'.format(np.shape(layer)) #(1, ?, ?, 18)
    return layer

def rpn_bbox_layer(layer , n_anchors =9):
    with tf.variable_scope('bbox'):
        layer = convolution2d('rpn_bbox_conv' ,layer, out_ch= n_anchors*4 , k=1 , act=None , s=1)
        layer  = tf.identity(layer , name='cls_output')
        print '** cls layer shape : {}'.format(np.shape(layer)) #(1, ?, ?, 18)
    return layer

def rpn_cls_loss(rpn_cls_score , rpn_labels):
    """
    :param rpn_cls_score : 1.shape : (1 , h , w , 18)
    :param rpn_labels:  1.shape : # (1 ,1 , h ,w )
    :return:
    """
    with tf.variable_scope('rpn_cls'):
        shape = tf.shape(rpn_cls_score)
        # (1, h, w, 18) ==> (1 , 18 , h , w)
        rpn_cls_score = tf.transpose(rpn_cls_score, [0, 3, 1, 2])
        # (1 , 18 , h , w) ==> (1 , 2 , h*9 , w)
        rpn_cls_score = tf.reshape(rpn_cls_score, [shape[0], 2, shape[3] // 2 * shape[1], shape[2]])
        # (1 , 2 , h*9 , w) ==> (1 , 2 , h*9 , w)
        rpn_cls_score = tf.transpose(rpn_cls_score, [0, 2, 3, 1])
        # (1 , h*9 , w , 2) ==> (h*9*w , 2 )
        rpn_cls_score = tf.reshape(rpn_cls_score, [-1, 2])
        # (1, h ,w ,1) ==> (1, 1, h, w)
        rpn_labels = tf.transpose(rpn_labels, [0, 2, 3, 1])
        rpn_labels = tf.reshape(rpn_labels, [-1])
        #
        cls_indices = tf.gather(rpn_cls_score, tf.where(tf.not_equal(rpn_labels, -1)), name='cls_indices')
        lab_indices = tf.gather(rpn_labels, tf.where(tf.not_equal(rpn_labels, -1)), name='lab_indices')
        rpn_cls_score = tf.reshape(cls_indices, [-1, 2])
        rpn_labels = tf.reshape(lab_indices, [-1])
        # Cross Entropy loss
        rpn_cross_entropy = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(logits=rpn_cls_score, labels=rpn_labels) , name='loss')
    return rpn_cross_entropy ,rpn_cls_score

def rpn_bbox_loss(rpn_bbox_pred, rpn_bbox_targets, rpn_inside_weights, rpn_outside_weights , rpn_labels):
    """

    :param rpn_bbox_pred: 1. shape : (1, h, w, 36)
    :param rpn_bbox_targets: 1.shape : (1, h, w, 36)
    :param rpn_inside_weights: 1.shape : (1, h, w, 36)
    :param rpn_outside_weights: 1.shape : (1, h, w, 36)
    :param rpn_labels: 1.shape : (1, h, w, 1)
    :return:
    """
    RPN_BBOX_LAMBDA = 10.0
    with tf.variable_scope('rpn_bbox'):
        # labels
        rpn_labels = tf.transpose(rpn_labels, [0, 2, 3, 1])
        rpn_labels = tf.reshape(rpn_labels, [-1])
        indices = tf.where(tf.not_equal(rpn_labels, -1))
        indices = tf.reshape(indices, shape=[-1])
        # RPN BBOX PREDICTION
        shape = tf.shape(rpn_bbox_pred)
        rpn_bbox_pred = tf.transpose(rpn_bbox_pred, [0, 3, 1, 2])
        rpn_bbox_pred = tf.reshape(rpn_bbox_pred, [shape[0], 4, shape[3] // 4 * shape[1], shape[2]])
        rpn_bbox_pred = tf.transpose(rpn_bbox_pred, [0, 2, 3, 1])
        rpn_bbox_pred = tf.reshape(rpn_bbox_pred, [-1, 4])
        rpn_bbox_pred_inds  = tf.gather(rpn_bbox_pred , indices)
        rpn_bbox_pred_inds = tf.reshape(rpn_bbox_pred_inds , shape=[-1 , 4] , name='rpn_bbox_pred')

        # RPN BBOX Target
        rpn_bbox_targets_inds  = tf.gather(rpn_bbox_targets , indices)
        rpn_bbox_targets_inds = tf.reshape(rpn_bbox_targets_inds , shape=[-1 , 4] ,name='rpn_bbox_targets')

        # RPN INSIDE WEIGHT
        rpn_inside_weights_inds  = tf.gather(rpn_inside_weights , indices )
        rpn_inside_weights_inds = tf.reshape(rpn_inside_weights_inds , shape=[-1 , 4] ,name='rpn_inside_weights')

        # RPN OUTSIDE WEIGHT
        rpn_outside_weights_inds = tf.gather(rpn_outside_weights, indices)
        rpn_outside_weights_inds = tf.reshape(rpn_outside_weights_inds, shape=[-1, 4] , name='rpn_outside_weights')

        loss = tf.multiply(rpn_inside_weights_inds, rpn_bbox_pred_inds - rpn_bbox_targets_inds)
        loss = smoothL1(loss, 3.0)
        #
        rpn_bbox_reg = tf.reduce_sum(tf.multiply(rpn_outside_weights_inds, loss))

        # Constant for weighting bounding box loss with classification loss
        rpn_bbox_reg = RPN_BBOX_LAMBDA * rpn_bbox_reg
        rpn_bbox_reg = tf.identity(rpn_bbox_reg , 'loss')
    return rpn_bbox_reg

def smoothL1(x, sigma):
    '''
    Tensorflow implementation of smooth L1 loss defined in Fast RCNN:
        (https://arxiv.org/pdf/1504.08083v2.pdf)
                    0.5 * (sigma * x)^2         if |x| < 1/sigma^2
    smoothL1(x) = {
                    |x| - 0.5/sigma^2           otherwise
    '''
    with tf.variable_scope('smoothL1'):
        conditional = tf.less(tf.abs(x), 1 / sigma ** 2)
        close = 0.5 * (sigma * x) ** 2
        far = tf.abs(x) - 0.5 / sigma ** 2
    return tf.where(conditional, close, far)

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