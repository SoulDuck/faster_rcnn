#-*- coding:utf-8 -*-
from bbox_overlaps import bbox_overlaps
import numpy as np
class Eval():
    def __init__(self):
        pass;
    def __call__(self, *args, **kwargs):
        pass;

    @classmethod
    def get_accuracy(cls , pred_bboxes , true_bboxes ):
        overlaps = bbox_overlaps(np.ascontiguousarray(pred_bboxes, np.float),
                                 np.ascontiguousarray(true_bboxes, np.float))
        maximun = np.max(overlaps , axis=1)
        true_conut = np.sum([maximun > 0.5])
        n_samples = len(maximun)
        acc = true_conut / float(n_samples)
        return acc

    @classmethod
    def get_accuracy_all(self , preds, trues ,n_classes):
        assert np.ndim(preds) == np.ndim(trues) and np.ndim(preds) == 2,\
            'preds n dim :{} trues n dims :{}'.format(np.ndim(preds),np.ndim(trues))
        assert np.shape(preds)[-1] == np.shape(trues)[-1] and np.shape(preds)[-1] == 5
        preds, trues = map(np.asarray ,[preds , trues])
        # preds [[x1,y1,x2,y2,label ]]
        # trues [[x1,y1,x2,y2,label ]]
        preds_cls , trues_cls = map(lambda x : np.asarray(x) , [preds , trues])
        acc = {}
        for i in range(1,n_classes):
            # background 는 계산하지 않는다
            preds_indices, trues_indices = map(lambda x: np.where([x == i])[1], [preds_cls, trues_cls])
            # pred 갯수가 0개이면 accuracy 가 0 이고 pred 갯수가 0 , trues =0 이면 pass 한다 .
            # trues =0 인데 preds갯수가 0이 아니면 acc 가 0 입니다.
            if len(preds_indices) == 0 and len(trues_indices) ==0:
                continue;
            elif len(trues_indices) ==0 and len(preds_indices) != 0:
                acc[i] = 0
                continue;
            elif len(preds_indices) ==0 and len(trues_indices) != 0:
                acc[i] = 0
                continue;
            # pick specific preds , trues

            picked_preds = preds[preds_indices]
            picked_trues = trues[trues_indices]

            # accuracy
            acc[i] = self.get_accuracy(picked_preds, picked_trues)
        return acc



    @classmethod
    def merge_acc(cls , acc_dict , dict_1):
        for key in dict_1.keys():
            acc = dict_1[key]

        if not key in acc_dict.keys():
            # new
            acc_dict[key] = [1,acc]
        else:
            # add elements
            # count
            acc_dict[key][0] += 1
            # count
            acc_dict[key][1] += acc

        return acc_dict

    @classmethod
    def get_meanacc(cls , acc_dict):
        mean_acc_list= []
        for key in acc_dict:
            mean_acc_list.append(acc_dict[key][1]/float(acc_dict[key][0]))
        return mean_acc_list

if __name__ == '__main__':
    preds= [[100,100,200,200,3] , [50,50,100,100,1]  ,[100,100,100,200,1] ,[300,200,400,400,2] ]
    trues = [[100,100,200,200,3] , [100,100,300,300,1] , [100,100,400,400,2] ,[50,70,400,400,2]]
    """
    Example 
    [[  1.00000000e+00   2.52493750e-01   1.12592576e-01]
    [7.81188970e-05   2.32552731e-05   1.07294986e-05]]
    """

    acc = Eval.get_accuracy_all(preds , trues , 4)
    print acc
    print np.mean(acc)




