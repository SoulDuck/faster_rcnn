#-*- coding:utf-8 -*-
import glob , os ,sys
from PIL import Image
import numpy as np
import numpy.random as npr
from utils import progress
from utils import progress
class Dataprovider():
    def __init__(self ,imgdir ,imgext):
        self.imgdir = imgdir
        self.imgext = imgext # image extension
        self.img_paths = glob.glob(os.path.join(self.imgdir , '*.{}'.format(self.imgext)) )
        self.img_names = map(lambda path : os.path.split(path)[-1] , self.img_paths)
        print '# images : {}'.format(len(self.img_paths))
    @classmethod
    def image_normalize(self , img):
        return img/255.

    def read_images_on_RAM(self ):
        raise NotImplementedError

    def read_gtbboxes_onRAM(self , label_path):
        raise NotImplementedError

    @classmethod
    def next_batch(self , imgs , labs  , n_batch):
        indices = npr.randint(0, len(labs))
        if n_batch == 1:
            batch_xs = imgs[indices : indices +1]
            batch_ys = labs[indices]
        else:
            batch_xs = imgs[indices]
            batch_ys = labs[indices]

        return batch_xs , batch_ys
    def get_name(self , path ):
        name=os.path.split(path)[-1]
        return name



class Wally(Dataprovider):
    def __init__(self ,imgdir ,imgext):
        Dataprovider.__init__(self , imgdir ,imgext)
    def read_images_on_RAM(self , normalize):
        imgs = []
        for i,path in enumerate(self.img_paths):
            progress(i , len(self.img_paths))
            img=np.asarray(Image.open(path).convert('RGB'))
            if normalize:
                img = img/255.
            imgs.append(img)
        return imgs

    def read_test_images_on_RAM(self , imgdir , imgext , normalize):
        img_paths = glob.glob(os.path.join(imgdir, '*.{}'.format(imgext)))
        print '# images : {}'.format(len(img_paths))
        imgs = []
        for i,path in enumerate(img_paths):
            progress(i , len(img_paths))
            img=np.asarray(Image.open(path).convert('RGB'))
            if normalize:
                img = img/255.
            imgs.append(img)
        return imgs



    def read_gtbboxes_onRAM(self , label_path):

        f=open(label_path , 'r')
        lines = f.readlines()
        lines = lines[1:]
        elements={}
        for line in lines:
            fname , h, w, label , x1,y1,x2,y2 = line.split(',')
            x1,y1,x2,y2 = map(lambda ele : int(ele.strip()) , [x1,y1,x2,y2])
            if 'waldo' in label:
                label = 1

            elements[fname] = [[x1,y1,x2,y2,label]]

        ret_elements = []
        for name in self.img_names:
            ret_elements.append(elements[name])

        return ret_elements


        pass;
class PocKia(Dataprovider):
    def __init__(self,imgdir ,imgext):
        Dataprovider.__init__(self, imgdir, imgext)
        self.n_imgs = len(self.img_paths)
    def read_images_on_RAM(self):
        # Ram 에다 다 올릴수 없어서 한장씩 불러오는 코드를 짜야 합니다
        pass;
    def generate_index(self , ind):
        if ind is None:
            ind = npr.randint(0,self.n_imgs)
        else:
            ind = ind % self.n_imgs
        return ind
    def read_image(self , normalize , ind ):
        img_path = self.img_paths[ind]
        img_name = self.get_name(img_path)
        # read image
        img = np.asarray(Image.open(img_path).convert('RGB'))
        # normalize
        if normalize:
           img = img / 255.
        # reshape image to feed
        img = np.reshape(img , [1]+list(np.shape(img)))
        return img  , img_name

    def read_label(self , labels , ind):
        return labels[ind]

    def read_gtbboxes(self,label_path):
        f = open(label_path, 'r')
        lines = f.readlines()
        ret_gtbboxes = {}
        for l in lines:
            tmp_list = []
            elements = l.split(',')
            img_name = elements[0]
            n_labels = elements[1]

            for n in range(int(n_labels)):
                x, y, w, h, btn = elements[2 + n * 5:2 + (n + 1) * 5]
                btn = btn.replace('button', '')
                x, y, w, h, btn = map(int, [x, y, w, h, btn])
                tmp_list.append([x, y, w + x, h + y, btn])
            ret_gtbboxes[img_name] = np.asarray(tmp_list)
        assert len(ret_gtbboxes) == len(lines)
        return ret_gtbboxes

    def read_gtbboxes_onRAM(self ,label_path):
        # 불러온 순서에 맞게 path을 조정합니다
        gtbboxes_dict = self.read_gtbboxes(label_path)
        ret_gtbboxes=[]
        error_labels = []
        for name in self.img_names:
            try:
                gt_bboxes=gtbboxes_dict[name]
                ret_gtbboxes.append(gt_bboxes)
            except KeyError:
                error_labels.append(name)
        print 'Error list : {}'.format(error_labels)
        return np.asarray(ret_gtbboxes)

if __name__ == '__main__':
    # Wally Test
    """
    wally = Wally(imgdir='./WallyDataset/images' , imgext='jpg')
    gt_bboxes = wally.read_gtbboxes_onRAM('./WallyDataset/annotations/annotations.csv')
    print np.shape(gt_bboxes)
    """
    # POC KIA Test
    pockia = PocKia(imgdir='/Volumes/My Passport/data/kia/review_data/test', imgext='jpg')
    #
    label_path = 'labels.txt'
    ind = pockia.generate_index(None)
    img = pockia.read_image(True , ind )
    # label dict
    labels = pockia.read_gtbboxes_onRAM(label_path)
    # Choose one label correspond image
    label = pockia.read_label(labels , ind )

    print 'Image shape : {} , label shape {} '.format(np.shape(img) , np.shape(label))

    import matplotlib.pyplot as plt
    import cv2
    import utils
    img = np.squeeze(img)*255
    label = np.asarray(label)
    labels = label[:,-1]
    bboxes = label[:, :-1]
    utils.draw_bboxes(np.squeeze(img), bboxes , labels ,savepath='tmp.png')



