# POC KIA Dataset
"""
imgdir = './poc_data/Images'
imgext = 'jpg'
label_path = './poc_data/labels_training.txt'
"""

# Wally Dataset
train_imgdir = './WallyDataset/images'
train_imgext = 'jpg'
label_path =  './WallyDataset/annotations/annotations.csv'

test_imgdir ='./WallyDataset/eval_images'
test_imgext  = 'jpg'
n_classes = 1+1
anchor_scales = [24, 36, 50]



#
model_dir = './models'
log_dir = './logs'
eval_imgdir ='./evalImage'
#
max_iter = 100000