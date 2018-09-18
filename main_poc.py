#-*- coding:utf-8 -*-
import tensorflow as tf
import numpy as np
import os
from anchor_target import anchor_target
from backbone import Backbone
from rpn import rpn_cls_loss ,rpn_bbox_loss
from proposal_target_layer import proposal_target_layer , inv_targets
from fast_rcnn import fast_rcnn , fast_rcnn_bbox_loss , fast_rcnn_cls_loss , get_interest_target
from rpn import rpn_cls_layer , rpn_bbox_layer
from utils import param_count
import roi
import sys , time
from utils import sess_start , optimizer , progress , draw_bboxes , draw_rectangles
from eval import Eval
from train import Train
from Dataprovider import PocKia
import configure as cfg
import utils
from nms import non_maximum_supression
# Configure
n_classes = cfg.n_classes
anchor_scales = cfg.anchor_scales

# Load Data

# Placeholder
x_ = tf.placeholder(dtype=tf.float32, shape=[1, None, None, 3], name='x_')
im_dims = tf.placeholder(tf.int32, [None, 2], name='im_dims')
gt_boxes = tf.placeholder(tf.int32, [None, 5], name='gt_boxes')
phase_train = tf.placeholder(tf.bool, name='phase_train')

# Backbone 수정
backbone = Backbone(x_,backbone_name='simple_convnet')
topconv_shape =tf.shape(backbone.top_conv)
top_n= topconv_shape[0]
top_h= topconv_shape[1]
top_w= topconv_shape[2]
feat_stride = backbone.feat_stride
top_conv = backbone.top_conv

# compute RPN target
# Output (h*w*9 , 4)
rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights = anchor_target(topconv_h=top_h, topconv_w=top_w,
                                                                                    gt_boxes=gt_boxes,
                                                                                    im_dims=im_dims,
                                                                                    _feat_stride=feat_stride,
                                                                                    anchor_scales=anchor_scales)
# (h*w*9 , 4) ==> (1 , h, w , 4*9)
# RPN CLS
rpn_cls = rpn_cls_layer(top_conv)
# RPN BBOX
rpn_bbox_pred = rpn_bbox_layer(top_conv)
# RPN CLS Loss
rpn_cls_loss_op ,A_op  = rpn_cls_loss(rpn_cls , rpn_labels) # rpn_labels_op 1 ,1 h ,w
# RPN BBOX Loss
rpn_bbox_loss_op = rpn_bbox_loss(rpn_bbox_pred ,rpn_bbox_targets , rpn_bbox_inside_weights , rpn_bbox_outside_weights , rpn_labels)
# 모든 roi 중에 foreground 와 background 을 뽑아 Fast RCNN 으로 넘겨준다
roi_blobs_op, roi_scores_op = \
    roi.roi_proposal(rpn_cls , rpn_bbox_pred , im_dims , feat_stride , anchor_scales ,is_training=phase_train)
# proposal target bboxes
ptl_rois_op, ptl_labels_op, ptl_bbox_targets_op, ptl_bbox_inside_weights_op, ptl_bbox_outside_weights_op = \
    proposal_target_layer(roi_blobs_op , gt_boxes , _num_classes= n_classes ) # ptl = Proposal Target Layer
rois_op = tf.cond(phase_train , lambda : ptl_rois_op , lambda : roi_blobs_op)
#

# Fast RCNN
fast_rcnn_cls_logits_op , fast_rcnn_bbox_logits_op = \
    fast_rcnn(top_conv , rois_op ,im_dims  , num_classes=n_classes , phase_train = phase_train)
#
itr_fr_bbox_target_op = get_interest_target(tf.argmax(fast_rcnn_cls_logits_op , axis =1), fast_rcnn_bbox_logits_op , n_classes )
# inverse blobs to coordinates (x1,y1,x2,y2)
itr_fr_blobs_op = inv_targets(rois_op, itr_fr_bbox_target_op)

# FastRCNN CLS Loss
fr_cls_loss_op = fast_rcnn_cls_loss(fast_rcnn_cls_logits_op , ptl_labels_op)
# FastRCNN BBOX Loss
fr_bbox_loss_op = fast_rcnn_bbox_loss(fast_rcnn_bbox_logits_op ,ptl_bbox_targets_op , ptl_bbox_inside_weights_op , ptl_bbox_outside_weights_op )
# if cls 1 , bbox = [2,3,1,3 ,4,5,6,7, 1,2,3,4] ==> [1,2,3,4]



# Loss
rpn_cost_op = rpn_cls_loss_op + rpn_bbox_loss_op
fr_cost_op = fr_cls_loss_op + fr_bbox_loss_op
cost_op = rpn_cost_op  + fr_cost_op
# Optimizer
train_op = optimizer(cost_op , 0.0001)
# Start Session
sess = sess_start()
param_count()
# Saver
saver = tf.train.Saver(max_to_keep=10)
# Restore Models
model_dir =  cfg.model_dir
if not os.path.isdir(model_dir):
    os.makedirs(model_dir)
if not tf.train.latest_checkpoint(model_dir) is None:
    print '{} restored '.format(tf.train.latest_checkpoint(model_dir))
    saver.restore(sess , tf.train.latest_checkpoint(model_dir))

# Write log
tb_writer = tf.summary.FileWriter(cfg.log_dir)
tb_writer.add_graph(tf.get_default_graph())

# Set feed
min_cost = 100000
ckpt = 100
eval_root_imgdir = cfg.eval_imgdir

# Load Dataset
train_imgdir = cfg.train_imgdir
train_imgext = cfg.train_imgext
test_imgdir = cfg.test_imgdir
test_imgext = cfg.test_imgext

train_label_path = cfg.train_label_path
test_label_path = cfg.test_label_path

pockia_train = PocKia(train_imgdir, train_imgext)
pockia_test = PocKia(test_imgdir, test_imgext)
# Load Labels
train_labels = pockia_train.read_gtbboxes(train_label_path)
test_labels = pockia_test.read_gtbboxes(test_label_path)

for i in range(cfg.max_iter):
    ind = pockia_train.generate_index(None)
    batch_xs , batch_names = pockia_train.read_image(True, ind)
    batch_ys = train_labels[batch_names]

    progress(i ,cfg.max_iter )
    # check normalize
    assert np.max(batch_xs) <= 1 ,'image max : {}'.format(np.max(batch_xs))

    # Image shape
    _ , h,w,ch = np.shape(batch_xs)
    # Set Train Feed
    train_feed = {x_ : batch_xs , gt_boxes:batch_ys , im_dims : np.asarray([[h,w]])  ,phase_train : True }
    # Set Train fetches
    train_fetches = [train_op , cost_op ,itr_fr_bbox_target_op ]
    # Training
    train , cost ,itr_fr_bbox_target = sess.run(train_fetches , train_feed)
    if i % ckpt  == 0 :
        print "train cost : {} \n".format(cost)
        # make folder
        eval_imgdir =os.path.join(eval_root_imgdir ,  str(i))
        if not os.path.isdir(eval_imgdir):
            os.makedirs(eval_imgdir)


        val_acc_counter = {} # (count , acc )
        merged_acc = {}
        for test_ind in range(len(pockia_test.img_paths)):
            utils.progress( test_ind , len(pockia_test.img_paths))
            # Get batch
            batch_xs , batch_names = pockia_test.read_image(True, test_ind)
            batch_ys = test_labels[batch_names]
            # Test Eval feed , fetches
            eval_feed = {x_: batch_xs, gt_boxes: batch_ys, im_dims: [[h, w]], phase_train: False}
            eval_fetches = [fast_rcnn_cls_logits_op, itr_fr_blobs_op]
            # Run sess
            cls_logits , itr_fr_blobs = sess.run(eval_fetches, eval_feed)
            cls_logits = np.argmax(cls_logits, axis=1)
            # (1,?, 5 ) ==> (?,5)
            itr_fr_blobs = np.squeeze(itr_fr_blobs )
            fr_blobs_cls = np.hstack([itr_fr_blobs , cls_logits.reshape([-1,1])])
            # NMS
            nms_keep = non_maximum_supression(fr_blobs_cls, 0.01)
            print 'before nms {} ==> after nms {}'.format(len(fr_blobs_cls), len(nms_keep))
            # Eval
            acc = Eval.get_accuracy_all(fr_blobs_cls[nms_keep] , batch_ys, n_classes=cfg.n_classes)
            # merge

            merged_acc = Eval.merge_acc(merged_acc , acc)
            print merged_acc


            # (height,width,3) ==>(height ,width,3)
            batch_xs = batch_xs.reshape(np.shape(batch_xs)[1:])
            # Draw Foreground Rectangle and Background Rectangle
            draw_rectangles(batch_xs*255, cls_logits, itr_fr_blobs,
                            savepath=os.path.join(eval_imgdir , '{}.jpg'.format(test_ind)))