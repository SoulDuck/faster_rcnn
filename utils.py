#-*- coding:utf-8 -*-
import sys ,os
import tensorflow as tf
from tensorflow.python.tools import inspect_checkpoint as chkp
import cv2
import numpy as np
def sess_start():
    # 필요한 만큼의 Gpu만 사용하게 하기
    sess=tf.Session()
    init=tf.group( tf.global_variables_initializer() , tf.local_variables_initializer() )
    sess.run(init)
    return sess

def optimizer(cost , lr):
    train_op= tf.train.AdamOptimizer(learning_rate=lr).minimize(cost)
    return train_op


def progress(i, max_step):
    msg = '\r {} / {}'.format(i, max_step)
    sys.stdout.write(msg)
    sys.stdout.flush()

def param_count():
    total_parameters = 0
    for variable in tf.trainable_variables():
        # shape is an array of tf.Dimension
        shape = variable.get_shape()
        variable_parameters = 1
        for dim in shape:
            variable_parameters *= dim.value
        total_parameters += variable_parameters
    print(total_parameters)


def draw_bboxes(img , bboxes, box_names  , savepath ):
    """
    :param bboxes x1,y1,x2,y2:
    :return:
    """
    for ind , box in enumerate(bboxes):
        x1, y1, x2, y2 = box  # x1 ,y1 ,x2 ,y2


        img = cv2.rectangle(img,(x1, y1),(x2 ,y2), (0,255,0),3)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img,str(box_names[ind]) , (x1,y1) , font , 1 , (255,255,255) ,2  ,cv2.LINE_AA)
    cv2.imwrite(savepath , img )


def draw_rectangles(img ,labels , bboxes , savepath ):

    # extract Indices
    bg_indices = np.where([labels == 0])[-1]
    fg_indices = np.where([labels != 0])[-1]
    bg_bboxes = bboxes[bg_indices]
    fg_bboxes = bboxes[fg_indices]
    fg_cls = labels[fg_indices]
    bg_cls = labels[bg_indices]

    # setting savepath of foreground and background
    savename = os.path.split(savepath)[1]
    savename = os.path.splitext(savename)[0]
    fg_savepath =savepath.replace(savename , savename+'_fg')
    bg_savepath = savepath.replace(savename, savename + '_bg')



    if not len(fg_bboxes) ==0:
        draw_bboxes(img , fg_bboxes , box_names=fg_cls , savepath= fg_savepath)
    # Draw Background BoundBox
    if not len(bg_bboxes) == 0:
        draw_bboxes(img, bg_bboxes, box_names= bg_cls ,savepath=bg_savepath)

if __name__ == '__main__':
    img_path ='./WallyDataset/eval_images/1.jpg'
