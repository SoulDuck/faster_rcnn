# POC KIA Dataset




train_imgdir = '/Volumes/My Passport/data/kia/review_data/training'
train_imgext = 'jpg'
test_imgdir = '/Volumes/My Passport/data/kia/review_data/test'
test_imgext = 'jpg'

train_label_path = '/Volumes/My Passport/data/kia/review_data/training/labels.txt'
test_label_path = '/Volumes/My Passport/data/kia/review_data/test/labels.txt'

"""
# Wally Dataset
train_imgdir = './WallyDataset/images'
train_imgext = 'jpg'

test_imgdir ='./WallyDataset/eval_images'
test_imgext  = 'jpg'

label_path =  './WallyDataset/annotations/annotations.csv'
"""
#

n_classes = 8+1
anchor_scales = [24, 36, 50]



#
model_dir = './models'
log_dir = './logs'
eval_imgdir ='./evalImage'
#
max_iter = 100000