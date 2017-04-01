# -*- coding:utf-8 -*-
import abc
from random import sample
from math import exp, log
from gbdt.tree import construct_decision_tree


class ClassifyMethod(object):
    __metaclass__=abc.ABCMeta
    def __init__(self, n_classes):
        self.K = n_classes

    @abc.abstractmethod
    def compute_residual(self, dataset, subset, F):
        """计算残差"""

    @abc.abstractmethod
    def update_f_value(self, F, tree, leaf_nodes, subset, dataset, learn_rate, label=None):
        """更新F_{m-1}的值"""

    @abc.abstractmethod
    def initialize(self, F, dataset):
        """初始化F_{0}的值"""

    @abc.abstractmethod
    def update_ternimal_regions(self, targets, idset):
        """更新叶子节点的返回值 \gamma_jm
        targets[id]: y_i, label or value
        """


class Regression(ClassifyMethod):
    """用于回归的最小平方误差损失函数"""
    def __init__(self, n_classes):
        if n_classes != 1:
            raise ValueError("``n_classes`` must be 1 for regression but "
                             "was %r" % n_classes)
        super(Regression, self).__init__(n_classes)

    def compute_residual(self, dataset, subset, F):
        residual = {}
        for id in subset:
            y_i = dataset.get_instance(id)['label']
            residual[id] = y_i - F[id]
        return residual

    def update_f_value(self, F, tree, leaf_nodes, subset, dataset, learn_rate, label=None):
        data_idset = set(dataset.get_ids())
        subset = set(subset)
        for node in leaf_nodes:
            for id in node.get_idset():
                F[id] += learn_rate*node.get_predict_value()
        for id in data_idset-subset:
            F[id] += learn_rate*tree.get_predict_value(dataset.get_instance(id))

    def initialize(self, F, dataset):
        """初始化F0，我们可以用训练样本的所有值的平均值来初始化，为了方便，这里初始化为0.0"""
        ids = dataset.get_ids()
        for id in ids:
            F[id] = 0.0

    def update_ternimal_regions(self, targets, idset):
        sum1 = sum([targets[id] for id in idset])
        return sum1/len(idset)


class BinClassify(ClassifyMethod):
    """二元分类"""
    def __init__(self, n_classes):
        if n_classes != 2:
            raise ValueError("{0:s} requires 2 classes.".format(
                self.__class__.__name__))
        super(BinClassify, self).__init__(1)

    # F: F_m-1()
    def compute_residual(self, dataset, subset, F):
        residual = {}
        for id in subset:
            y_i = dataset.get_instance(id)['label']
            residual[id] = 2.0*y_i/(1+exp(2*y_i*F[id]))
        return residual

    def update_f_value(self, F, tree, leaf_nodes, subset, dataset, learn_rate, label=None):
        data_idset = set(dataset.get_ids())
        subset = set(subset)
        for node in leaf_nodes:
            for id in node.get_idset():
                F[id] += learn_rate*node.get_predict_value()
        for id in data_idset-subset:
            F[id] += learn_rate*tree.get_predict_value(dataset.get_instance(id))

    def initialize(self, F, dataset):
        ids = dataset.get_ids()
        for id in ids:
            F[id] = 0.0

    def update_ternimal_regions(self, targets, idset):
        sum1 = sum([targets[id] for id in idset])
        if sum1 == 0:
            return sum1
        sum2 = sum([abs(targets[id])*(2-abs(targets[id])) for id in idset])
        return sum1 / sum2


class KClassify(ClassifyMethod):
    """多元分类的损失函数"""
    def __init__(self, n_classes, labelset):
        self.labelset = set([label for label in labelset])
        if n_classes < 2:
            raise ValueError("{0:s} requires more than 1 classes.".format(
                self.__class__.__name__))
        super(KClassify, self).__init__(n_classes)

    def compute_residual(self, dataset, subset, F):
        residual = {}
        label_set = dataset.get_label_set()
        for id in subset:
            residual[id] = {}
            p_sum = sum([exp(F[id][x]) for x in label_set])
            # 对于同一样本在不同类别的残差，需要在同一次迭代中更新在不同类别的残差
            for label in label_set:
                p = exp(F[id][label])/p_sum
                if dataset.get_instance(id)["label"] == label:
                    y = 1.0
                else:
                    y = 0.0
                residual[id][label] = y-p
        return residual

    def update_f_value(self, F, tree, leaf_nodes, subset, dataset, learn_rate, label=None):
        data_idset = set(dataset.get_ids())
        subset = set(subset)
        for node in leaf_nodes:
            for id in node.get_idset():
                F[id][label] += learn_rate*node.get_predict_value()
        # 更新OOB的样本
        for id in data_idset-subset:
            F[id][label] += learn_rate*tree.get_predict_value(dataset.get_instance(id))

    def initialize(self, F, dataset):
        ids = dataset.get_ids()
        for id in ids:
            F[id] = dict()
            for label in dataset.get_label_set():
                F[id][label] = 0.0

    def update_ternimal_regions(self, targets, idset):
        sum1 = sum([targets[id] for id in idset])
        if sum1 == 0:
            return sum1
        sum2 = sum([abs(targets[id])*(1-abs(targets[id])) for id in idset])
        return ((self.K-1)/self.K)*(sum1/sum2)


class GBDT:
    def __init__(self, max_iter, sample_rate, learn_rate, max_depth, method_name='multi-classification', split_points=0):
        self.max_iter = max_iter
        self.sample_rate = sample_rate
        self.learn_rate = learn_rate
        self.max_depth = max_depth
        self.method_name = method_name
        self.split_points = split_points
        self.method = None
        self.Mtrees = dict()    # M weak trees

    def fit(self, dataset, train_data):
        if self.method_name == 'multi-classification':
            label_set = dataset.get_label_set()
            self.method = KClassify(dataset.get_label_size(), label_set)
            F = dict()  # 记录F_{m-1}的值
            self.method.initialize(F, dataset)
            for iter in range(self.max_iter):
                subset = train_data
                if 0 < self.sample_rate < 1:    # random sampling subset of training data, a SGD(Stochastic) method
                    subset = sample(subset, int(len(subset)*self.sample_rate))
                self.Mtrees[iter] = dict()
                # 用损失函数的负梯度作为回归问题提升树的残差近似值
                residual = self.method.compute_residual(dataset, subset, F)
                for label in label_set:
                    # 挂在叶子节点下的各种样本,只有到迭代的max-depth才会使用
                    # 存放的各个叶子节点，注意叶子节点存放的是各个条件下的样本集点
                    leaf_nodes = []
                    targets = {}
                    for id in subset:
                        targets[id] = residual[id][label]
                    # 对某一个具体的label-K分类，选择max-depth个特征构造决策树
                    tree = construct_decision_tree(dataset, subset, targets, 0, leaf_nodes, self.max_depth, self.method, self.split_points)
                    self.Mtrees[iter][label] = tree
                    self.method.update_f_value(F, tree, leaf_nodes, subset, dataset, self.learn_rate, label)
                train_loss = self.compute_loss(dataset, train_data, F)
                print("iter%d : average train_loss=%f" % (iter, train_loss))

        else:
            if self.method_name == 'binary-classification':
                self.method = BinClassify(n_classes=dataset.get_label_size())
            elif self.method_name == 'regression':
                self.method = Regression(n_classes=1)

            F = dict()  # 记录F_{m-1}的值
            self.method.initialize(F, dataset)
            for iter in range(self.max_iter):
                subset = train_data
                if 0 < self.sample_rate < 1:
                    subset = sample(subset, int(len(subset)*self.sample_rate))
                # 用损失函数的负梯度作为回归问题提升树的残差近似值
                residual = self.method.compute_residual(dataset, subset, F)
                leaf_nodes = []
                targets = residual
                tree = construct_decision_tree(dataset, subset, targets, 0, leaf_nodes, self.max_depth, self.method, self.split_points)
                self.Mtrees[iter] = tree
                self.method.update_f_value(F, tree, leaf_nodes, subset, dataset, self.learn_rate)
                train_loss = self.compute_loss(dataset, train_data, F)
                print("iter%d : train loss=%f" % (iter,train_loss))
                #print self.Mtrees[iter]

    def compute_loss(self, dataset, subset, F):
        loss = 0.0
        if self.method.K == 1:  # regressing
            for id in dataset.get_ids():
                y_i = dataset.get_instance(id)['label']
                f_value = F[id]
                p_1 = 1/(1+exp(-2*f_value))
                try:
                    loss -= ((1+y_i)*log(p_1)/2) + ((1-y_i)*log(1-p_1)/2)
                except ValueError:
                    print(y_i, p_1)
        else:
            for id in dataset.get_ids():
                instance = dataset.get_instance(id)
                f_values = F[id]    #[0,0,1,0,...]
                exp_values = {}
                for label in f_values:
                    exp_values[label] = exp(f_values[label])
                probs = {}
                for label in f_values:
                    probs[label] = exp_values[label]/sum(exp_values.values())
                    # 预测的越准确则log(probs[instance["label"]])越接近0 loss也就越小
                loss -= log(probs[instance["label"]])
        return loss/dataset.size()

    def compute_F(self, instance):
        """计算样本的F值"""
        if self.method.K == 1:
            f_value = 0.0
            for iter in self.Mtrees:
                f_value += self.learn_rate * self.Mtrees[iter].get_predict_value(instance)
        else:
            f_value = dict()
            for label in self.method.labelset:
                f_value[label] = 0.0
            for iter in self.Mtrees:
                # 对于多分类问题，为每个类别构造一颗回归树
                for label in self.method.labelset:
                    tree = self.Mtrees[iter]
                    f_value[label] += self.learn_rate * tree.get_predict_value(instance)
        return f_value

    def predict(self, instance):
        """
        对于回归和二元分类返回F值
        对于多元分类返回每一类的F值
        """
        return self.compute_F(instance)

    def predict_prob(self, instance):
        """为了统一二元分类和多元分类，返回属于每个类别的概率"""
        if self.method.K == 1:
            f_value = self.compute_F(instance)
            probs = dict()
            probs['+1'] = 1/(1+exp(-2*f_value))
            probs['-1'] = 1 - probs['+1']
        else:
            f_value = self.compute_F(instance)
            exp_values = dict()
            for label in f_value:
                exp_values[label] = exp(f_value[label])
            exp_sum = sum(exp_values.values())
            probs = dict()
            # 归一化，并得到相应的概率值
            for label in exp_values:
                probs[label] = exp_values[label]/exp_sum
        return probs

    def predict_label(self, instance):
        """预测标签"""
        probs = self.predict_prob(instance)
        max_v = [-1, -1]
        for k,v in probs.iteritems():
            if v > max_v[1]:
                max_v = [k, v]
        return max_v[0]

