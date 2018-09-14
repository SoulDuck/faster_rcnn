#-*- coding:utf-8 -*-
import sys
import tensorflow as tf
from tensorflow.python.tools import inspect_checkpoint as chkp
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