# POC KIA Dataset




train_imgdir = './clutteredPOCKIA_TRAIN/Images'
train_imgext = 'jpg'
test_imgdir = './clutteredPOCKIA_TEST_2/Images'
test_imgext = 'jpg'

train_label_path = './clutteredPOCKIA_TRAIN/poc_labels.txt'
test_label_path = './clutteredPOCKIA_TEST_2/poc_labels.txt'

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