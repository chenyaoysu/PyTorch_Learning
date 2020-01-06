# /usr/bin/env python
# -*- coding: utf-8 -*-

"""
    描述: PyTorch 实现 k-近邻(kNN) 算法。
    Author: Chenyyx
    Date: 2019-12-18
"""

# 导入必要的库
import time
import matplotlib.pyplot as plt

import torch as t
import numpy as np
from collections import Counter


# 手动设置种子
t.manual_seed(2019)


# ====== 2.1 - Python 版本 knn 算法的实现 start =======
# 创建我们需要用到的数据集和标签
def create_dataset():
    """
    创建数据集和数据样本对应的标签
    """
    features = np.array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return features, labels


# knn 算法
def knn_python(input_x, dataset, labels, k):
    """
    :param input_x: 待分类的输入向量
    :param dataset: 作为参考计算距离的训练样本集
    :param labels: 数据样本对应的分类标签
    :param k: 选择最近邻样本的数目
    """
    # 1. 计算待测样本与参考样本之间的欧式距离
    dist = np.sum((input_x - dataset) ** 2, axis=1) ** 0.5

    # 2. 选取 k 个最近邻样本的标签
    k_labels = [labels[index] for index in dist.argsort()[0: k]]

    # 3. 得到出现次数最多的标签作为最终的分类类别
    label = Counter(k_labels).most_common(1)[0][0]
    return label


# 测试 python 版本 knn
def test_python():
    # 调用测试
    features, labels = create_dataset()

    # # 将数据在图上展示出来，查看一下
    # plt.plot(features[:2, 0], features[:2, 1], 'bo', label='A')
    # plt.plot(features[2:, 0], features[2:, 1], 'ro', label='B')
    # plt.legend()
    # plt.show()

    # 设置 k
    k = 3

    # 构造测试数据
    input_x = np.array([0.1, 0.1])

    # 调用 knn 算法
    result = knn_python(input_x, features, labels, k)

    # 输出结果
    print(result)
# ====== 2.1 - Python 版本 knn 算法的实现 end =======


# ====== 2.2 - PyTorch 版本 knn 算法的实现 start =======
def knn_pytorch(ref, qry, lab, k):
    """
    PyTorch 版本的 knn 实现
    :param ref: 训练样本数据
    :param qry: 输入数据
    :param lab: 训练样本对应的labels
    :param k: k 个最近邻样本
    """
    n, d = ref.size()
    m, d = qry.size()
    mref = ref.expand(m, n, d)
    mqry = qry.expand(n, m, d).transpose(0, 1)

    # 计算对应的距离以及对应的索引
    dist = t.sum((mqry - mref) ** 2, 2).squeeze() ** 0.5
    dist, indx = t.topk(dist, k, dim=1, largest=False, sorted=False)

    # 最终要返回的结果
    k_labs = []
    for num_k in indx:
        # 获取相对应的 k 个 labels
        k_lab = [lab[index] for index in num_k]
        # 得到出现次数最多的标签作为最终的分类类别
        label = Counter(k_lab).most_common(1)[0][0]
        k_labs.append(label)
    return k_labs


# 测试
def test_pytorch():
    ref = t.Tensor(500, 2).random_()
    qry = t.Tensor(5, 2).random_()

    lab = ['A'] * 250
    lab.extend(['B'] * 250)  # lab = ['A', 'A', ..., 'B', 'B', ...]

    # 设置 k
    k = 7

    # # 将测试使用的数据在图上画出来
    # plt.plot(ref[:250, 0].data.numpy(), ref[:250, 1].data.numpy(), 'bo', label='A')
    # plt.plot(ref[250:, 0].data.numpy(), ref[250:, 1].data.numpy(), 'ro', label='B')
    # plt.plot(qry[:, 0].data.numpy(), qry[:, 1].data.numpy(), 'go', label='qry')
    # plt.legend()
    t0 = time.time()
    result_label = knn_pytorch(ref, qry, lab, k)

    print(result_label)
    print('cost_time -----', time.time() - t0)
# ====== 2.2 - PyTorch 版本 knn 算法的实现 end ==========


# ====== 2.3 - scikit-learn 版本 knn 算法的实现 start =======
# 导入相关的库
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets


# 测试
def test_sklearn():
    # 预处理数据
    # 导入鸢尾花数据集
    iris = datasets.load_iris()

    # 因为鸢尾花数据集的特征很多，我们测试只使用前两个特征作为数据集
    X = iris.data[:, :2]
    y = iris.target

    # 步长
    h = 0.02

    # 设置 k
    k = 15

    # 设置 weights
    weights = 'distance'

    # 创建 color maps
    cmap_light = ListedColormap(['orange', 'cyan', 'cornflowerblue'])
    cmap_bold = ListedColormap(['darkorange', 'c', 'darkblue'])

    # 创建 knn 分类算法实例
    clf = neighbors.KNeighborsClassifier(k, weights=weights)
    # 拟合数据
    clf.fit(X, y)

    # 在图中画出每个点的边界
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    # # 在图上将结果展示出来
    # Z = Z.reshape(xx.shape)
    # plt.figure()
    # plt.pcolormesh(xx, yy, Z, cmap=cmap_light)
    #
    # # 训练数据也画出来
    # plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold, edgecolor='k', s=20)
    # plt.xlim(xx.min(), xx.max())
    # plt.ylim(yy.min(), yy.max())
    # plt.title("3-Class classification (k = %i, weights = '%s')" % (k, weights))
    # plt.show()

# ====== 2.3 - scikit-learn 版本 knn 算法的实现 end =========


if __name__ == '__main__':
    # 测试 python 版本的 knn
    test_python()

    # # 测试 PyTorch 版本的 knn
    # test_pytorch()
    #
    # # 测试 sklearn 版本的 knn
    # test_sklearn()
