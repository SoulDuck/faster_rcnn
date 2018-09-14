import glob , os ,sys
from PIL import Image
import numpy as np
import numpy.random as npr
from utils import progress
class Dataprovider():
    def __init__(self ,imgdir ,imgext):
        self.imgdir = imgdir
        self.imgext = imgext # image extension
        self.img_paths = glob.glob(os.path.join(self.imgdir , '*.{}'.format(self.imgext)) )
        self.img_names = map(lambda path : os.path.split(path)[-1] , self.img_paths)
        print '# images : {}'.format(len(self.img_paths))

    @classmethod
    def images_normalize(self , imgs):
        return imgs/255.


    def read_imagjes_on_RAM(self):
        print "{} images  loaded on RAM".format(len(self.img_paths))
        imgs=[]

        for i,path in enumerate(self.img_paths):
            try:
                progress(i,len(self.img_paths))
                img=np.asarray(Image.open(path).convert('RGB'))
                img = img/255.
                imgs.append(img)

            except:
                continue;

        imgs = np.asarray(imgs)
        print
        return imgs

    def _read_gtbboxes(self,label_path):
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

    def read_gtbboxes_onRAM(self,label_path):
        gtbboxes_dict = self._read_gtbboxes(label_path)
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

if __name__ == '__main__':
    dataprovider = Dataprovider(imgdir='/Users/seongjungkim/Desktop/test' , imgext='jpg')
    gt_bboxes = dataprovider.read_gtbboxes_onRAM('/Users/seongjungkim/Desktop/test/labels.txt')

    print np.shape(gt_bboxes)


