import tensorflow as tf
import numpy as np
from cnn import convolution2d
class Backbone(object):
    def __init__(self ,x_ , backbone_name):
        self.x_ = x_
        self.kernels = None
        self.out_channels = None
        self.strides = None
        self.backbone_name = backbone_name
        if backbone_name == 'simple_convnet':
            self.simple_convnet()
        else:
            raise AssertionError
        assert not self.top_conv is None and not self.feat_stride is None
    def simple_convnet(self ):
        print '############################################# '
        print '###### Convolution Network building....###### '
        print '############################################# '

        self.kernels=[5, 3, 3, 3, 3 ,3 ]
        self.out_channels=[16, 16, 32, 64, 64 , 128]
        self.strides = [2, 2, 2, 2 ,2 ,2 ]
        layer=self.x_
        assert len(self.kernels) == len(self.out_channels) == len(self.strides)
        for i in range(len(self.kernels)):
            layer = convolution2d(name='conv_{}'.format(i), x=layer, out_ch=self.out_channels[i], k=self.kernels[i],
                                  s=self.strides[i],
                                  padding='SAME')
        self.top_conv = tf.identity(layer , 'top_conv')
        self.feat_stride = np.prod(self.strides)



