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
        assert np.ndim(preds) == np.ndim(trues)  and np.ndim(preds) == 2
        assert np.shape(preds)[-1] == np.shape(trues)[-1] and np.shape(preds)[-1] == 5
        preds, trues = map(np.asarray ,[preds , trues])
        # preds [[x1,y1,x2,y2,label ]]
        # trues [[x1,y1,x2,y2,label ]]
        preds_cls , trues_cls = map(lambda x : np.asarray(x) , [preds , trues])
        acc = []
        for i in range(n_classes):
            preds_indices, trues_indices = map(lambda x: np.where([preds_cls == i])[1], [preds_cls, trues_cls])
            if len(preds_indices) == 0 and len(trues_indices) ==0:
                continue;
            elif len(trues_indices) ==0 and len(preds_indices) != 0:
                acc.append(0)
                continue;
            elif len(preds_indices) ==0 and len(trues_indices) != 0:
                acc.append(0)
                continue;
            # pick specific preds , trues
            picked_preds = preds[preds_indices]
            picked_trues = trues[trues_indices]

            # accuracy
            acc.append(self.get_accuracy(picked_preds, picked_trues))
        return acc








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




