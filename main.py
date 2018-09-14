#-*- coding:utf-8 -*-
import tensorflow as tf
import numpy as np
from anchor_target import anchor_target
from backbone import Backbone
from rpn import rpn_cls_loss ,rpn_bbox_loss
from proposal_target_layer import proposal_target_layer , inv_targets
from fast_rcnn import fast_rcnn , fast_rcnn_bbox_loss , fast_rcnn_cls_loss , get_interest_target
from rpn import rpn_cls_layer , rpn_bbox_layer
from utils import param_count
import roi
import sys , time
from utils import sess_start , optimizer
from eval import Eval
from train import Train
from Dataprovider import Dataprovider

# Configure
n_classes = 8+1
anchor_scales = [24, 36, 50]

# Load Data
imgdir = './samples'
imgext = 'jpg'
label_path = './samples/labels.txt'
dataprovider = Dataprovider(imgdir , imgext)
train_labs = dataprovider.read_gtbboxes_onRAM(label_path)
train_imgs = dataprovider.read_images_on_RAM()
train_imgs = dataprovider.images_normalize(train_imgs )




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
fast_rcnn_cls_logits , fast_rcnn_bbox_logits = \
    fast_rcnn(top_conv , rois_op ,im_dims  , num_classes=n_classes , phase_train = phase_train)
# FastRCNN CLS Loss
fr_cls_loss_op = fast_rcnn_cls_loss(fast_rcnn_cls_logits , ptl_labels_op)
# FastRCNN BBOX Loss
fr_bbox_loss_op = fast_rcnn_bbox_loss(fast_rcnn_bbox_logits ,ptl_bbox_targets_op , ptl_bbox_inside_weights_op , ptl_bbox_outside_weights_op )
# if cls 1 , bbox = [2,3,1,3 ,4,5,6,7, 1,2,3,4] ==> [1,2,3,4]
itr_fr_bbox_target_op = get_interest_target(tf.argmax(fast_rcnn_cls_logits , axis =1), fast_rcnn_bbox_logits , n_classes )
# inverse blobs to coordinates (x1,y1,x2,y2)
itr_fr_blobs_op = inv_targets(rois_op, itr_fr_bbox_target_op)



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
model_dir =  './models'
if not tf.train.latest_checkpoint(model_dir) is None:
    print '{} restored '.format(tf.train.latest_checkpoint(model_dir))
saver.restore(sess , tf.train.latest_checkpoint(model_dir))

# Write log
tb_writer = tf.summary.FileWriter('logs')
tb_writer.add_graph(tf.get_default_graph())

# Set feed
batch_xs , batch_ys = dataprovider.next_batch(train_imgs , train_labs , 1)
_ , h,w,ch = np.shape(batch_xs)

train_feed = {x_ : batch_xs , gt_boxes:batch_ys , im_dims : np.asarray([[h,w]])  ,phase_train : True }
eval_feed = {x_ : batch_xs , gt_boxes:batch_ys , im_dims : [[h,w]]  ,phase_train : False }

# Set fetches
eval_fetches = [cost_op , top_conv , itr_fr_blobs_op]
train_fetches = [train_op , cost_op ,itr_fr_bbox_target_op ]

# Training
train , cost ,itr_fr_bbox_target = sess.run(train_fetches , train_feed)
cost , top_conv , itr_fr_blobs = sess.run(eval_fetches, train_feed)