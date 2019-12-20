# /usr/bin/env python
# -*- coding: utf-8 -*-

"""
    描述: PyTorch 实现 线性回归(Linear Regression) 算法。
    Author: Chenyyx
    Date: 2019-12-18
"""

import numpy as np
import torch as t
import matplotlib.pyplot as plt

# 手动设置种子
t.manual_seed(2019)

# 读入训练数据 x 和 y
x_train = np.array([[3.3], [4.4], [5.5], [6.71], [6.93], [4.168],
                    [9.779], [6.182], [7.59], [2.167], [7.042],
                    [10.791], [5.313], [7.997], [3.1]], dtype=np.float32)

y_train = np.array([[1.7], [2.76], [2.09], [3.19], [1.694], [1.573],
                    [3.366], [2.596], [2.53], [1.221], [2.827],
                    [3.465], [1.65], [2.904], [1.3]], dtype=np.float32)

# # 将数据的图像画出来
# plt.figure(figsize=(12, 4), dpi=60)
# plt.plot(x_train, y_train, 'bo')
# plt.show()

# 将数据转换成 Tensor
x_train = t.from_numpy(x_train)
y_train = t.from_numpy(y_train)

# 定义参数 w 和 b
w = t.randn(1, requires_grad=True)
b = t.zeros(1, requires_grad=True)


# 构建线性回归模型
def my_linear_model(x):
    return x * w + b


# 这样就定义好了模型，在进行参数更新之前，我们可以先看看模型的输出结果长什么样
y_predict = my_linear_model(x_train)

# # 将模型的输出结果展示出来
# plt.plot(x_train.data.numpy(), y_train.data.numpy(), 'bo', label='real')
# plt.plot(x_train.data.numpy(), y_predict.data.numpy(), 'ro-', label='estimated')
# plt.legend()
# plt.show()


# 计算误差
def get_loss(y_predict, y_train):
    return t.mean(y_predict - y_train) ** 2


# loss = get_loss(y_predict, y_train)

# 打印一下 loss 的大小
# print(loss)

# # 自动求导
# loss.backward()
#
# # 查看 w 和 b 的梯度
# # print(w.grad)
# # print(b.grad)
#
# # 更新一次参数
# w.data = w.data - 1e-2 * w.grad.data
# b.data = b.data - 1e-2 * b.grad.data
#
# # 更新完参数之后，我们再看一下模型输出的结果
# y_predict = my_linear_model(x_train)

# plt.plot(x_train.data.numpy(), y_train.data.numpy(), 'bo', label='real')
# plt.plot(x_train.data.numpy(), y_predict.data.numpy(), 'ro-', label='estimated')
# plt.legend()
# plt.show()

# 如果希望能够比较好地拟合蓝色数据的真实值，那么我们就需要再进行几次简单地更新
epoch = 10

for e in range(epoch):
    y_ = my_linear_model(x_train)
    loss = get_loss(y_, y_train)

    loss.backward()  # 进行自动求导，反向传播

    w.data = w.data - 1e-2 * w.grad.data  # 更新 w
    b.data = b.data - 1e-2 * b.grad.data  # 更新 b
    print('epoch: {}, loss: {}'.format(e, type(loss)))
    w.grad.zero_()  # 将梯度归零
    b.grad.zero_()  # 将梯度归零

