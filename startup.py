# -*- coding:utf-8 -*-
__author__ = 'Dragonfly'
"""
如下为credit.data.csv文件的训练信息
iter1 : train loss=0.371342
iter2 : train loss=0.238326
iter3 : train loss=0.163624
iter4 : train loss=0.123063
iter5 : train loss=0.087872
iter6 : train loss=0.065684
iter7 : train loss=0.049936
iter8 : train loss=0.041866
iter9 : train loss=0.035695
iter10 : train loss=0.030581
iter11 : train loss=0.027034
iter12 : train loss=0.024570
iter13 : train loss=0.019227
iter14 : train loss=0.015794
iter15 : train loss=0.013484
iter16 : train loss=0.010941
iter17 : train loss=0.009879
iter18 : train loss=0.008619
iter19 : train loss=0.007306
iter20 : train loss=0.005610
"""
from gbdt.data import DataSet
from gbdt.model import GBDT
from random import sample

def evaluate(gbdt, subset, datas):
    c = 0
    for i in subset:
        y  = datas.get_instance(i)
        f = int(gbdt.predict_label(y))
        label = int(y['label'])
        if label != f:
            c += 1
    print 'error rate:', c*1.0 /len(subset)


if __name__ == '__main__':
    datas = DataSet('./data/credit.data.csv')
    gbdt = GBDT(max_iter=20, sample_rate=0.8, learn_rate=0.5, max_depth=7, method_name='binary-classification')
    ids = datas.get_ids()
    subset = sample(ids, int(len(ids)*0.2))
    gbdt.fit(datas, ids)
    evaluate(gbdt, subset, datas)
