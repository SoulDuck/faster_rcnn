# POC KIA Dataset
"""
imgdir = './poc_data/Images'
imgext = 'jpg'
label_path = './poc_data/labels_training.txt'
"""

# Wally Dataset
imgdir = 'WallyDataset/images'
imgext = 'jpg'
label_path =  'WallyDataset/annotations/annotations.csv'
n_classes = 8+1
anchor_scales = [24, 36, 50]

model_dir = './models'
log_dir = './logs'
#
max_iter = 100000