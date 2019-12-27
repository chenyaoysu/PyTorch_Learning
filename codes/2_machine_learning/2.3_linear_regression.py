# /usr/bin/env python
# -*- coding: utf-8 -*-

"""
    描述: PyTorch 实现 线性回归(Linear Regression) 算法。
    Author: Chenyyx
    Date: 2019-12-18
"""

# 导入必要的库
import numpy as np
import torch
import matplotlib.pyplot as plt

# 手动设置种子
torch.manual_seed(2019)

# ------------- 1 - 基础版 Linear Regression 实现 ---------------------

# 设置训练数据，包括 训练特征（x_train） 和 训练标签（y_train）
x_train = np.array([[3.51], [4.32], [5.8], [6.24], [7.9], [4.236],
                    [10.569], [7.158], [7.639], [1.134], [7.422],
                    [10.81], [6.021], [8.002], [2.958]], dtype=np.float32)
y_train = np.array([[1.732], [2.414], [2.039], [2.919], [1.594], [1.757],
                    [2.962], [2.561], [3.333], [1.222], [2.933],
                    [3.652], [2.865], [2.934], [1.314]], dtype=np.float32)

# # 将训练数据的图像画出来
# plt.plot(x_train, y_train, 'bo')
# plt.show()

# 将数据转换成 Tensor
x_train = torch.from_numpy(x_train)
y_train = torch.from_numpy(y_train)

# 定义参数 w 和 b
w = torch.randn(1, requires_grad=True)
b = torch.zeros(1, requires_grad=True)


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
    return torch.mean(y_predict - y_train) ** 2


# # 调用 get_loss 计算损失
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
# # 设置 学习率 learning_rate
# learning_rate = 1e-2
# w.data = w.data - learning_rate * w.grad.data
# b.data = b.data - learning_rate * b.grad.data
#
# # 更新完参数之后，我们再看一下模型输出的结果
# y_predict = my_linear_model(x_train)

# plt.plot(x_train.data.numpy(), y_train.data.numpy(), 'bo', label='real')
# plt.plot(x_train.data.numpy(), y_predict.data.numpy(), 'ro-', label='estimated')
# plt.legend()
# plt.show()

# 如果希望能够比较好地拟合蓝色数据的真实值，那么我们就需要再进行几次 w 和 b 参数的更新

# 在接下来的计算已经更新参数之前，需要将 w 和 b 参数的求导数据清零，因为之前计算过相应的导数，已经保存有数据了，不清零的话，会出现错误

w.grad.zero_()
b.grad.zero_()

# 设置更新次数
epoch = 200

for e in range(epoch):
    y_ = my_linear_model(x_train)
    loss = get_loss(y_, y_train)

    loss.backward()  # 进行自动求导，反向传播

    w.data = w.data - 1e-2 * w.grad.data  # 更新 w
    b.data = b.data - 1e-2 * b.grad.data  # 更新 b
    print('epoch: {}, loss: {}'.format(e, type(loss)))
    w.grad.zero_()  # 将梯度归零
    b.grad.zero_()  # 将梯度归零

# # 上面就训练完成了，接下来我们可以再次查看一下拟合效果
# y_ = my_linear_model(x_train)
# plt.plot(x_train.data.numpy(), y_train.data.numpy(), 'bo', label='real')
# plt.plot(x_train.data.numpy(), y_.data.numpy(), 'ro-', label='estimated')
# plt.legend()
# plt.show()

# --------------- 2 - 进阶版 Linear Regression 多项式回归模型 -------------

# 定义一个多项式函数
w_final = np.array([0.3, 2.8, 2.7])  # 定义参数
b_final = np.array([0.9])  # 定义参数

# 将多项式打印出来
multi_f = 'y = {:.2f} + {:.2f} * x + {:.2f} * x^2 + {:.2f} * x^3'.format(
    b_final[0], w_final[0], w_final[1], w_final[2])

# 将多项式公式打印出来
# print(multi_f)

# 将多项式的函数曲线画出来
x_sample = np.arange(-3, 2.9, 0.1)
y_sample = b_final[0] + w_final[0] * x_sample + w_final[1] * x_sample ** 2 + w_final[2] * x_sample ** 3

# plt.plot(x_sample, y_sample, label='real curve')
# plt.legend()
# plt.show()


# 创建数据
# x 是一个如下矩阵 [x, x^2, x^3]
# y 是函数的结果 [y]
x_train = np.stack([x_sample ** i for i in range(1, 4)], axis=1)
x_train = torch.from_numpy(x_train).float()  # 转换成 float tensor

y_train = torch.from_numpy(y_sample).float().unsqueeze(1)  # 转化成 float tensor

# 定义参数和模型
w = torch.randn(3, 1, requires_grad=True)
b = torch.zeros(1, requires_grad=True)


# # 将 x_train 和 y_train 转为 Tensor
# x_train = torch.Tensor(x_train)
# y_train = torch.Tensor(y_train)

# torch.mm 表示两个矩阵相乘
def my_multi_linear(x):
    return torch.mm(x, w) + b


# 定义 loss 函数
def get_multi_loss(y_pred, y):
    return torch.mean((y_pred - y_train) ** 2)


# # 画出更新之前的模型 和 真实数据 之间的对比
y_pred = my_multi_linear(x_train)
#
# plt.plot(x_train.data.numpy()[:, 0], y_pred.data.numpy(), label='fitting curve', color='r')
# plt.plot(x_train.data.numpy()[:, 0], y_sample, label='real curve', color='b')
# plt.legend()

# 计算误差
multi_loss = get_multi_loss(y_pred, y_train)
# print(multi_loss)

# 自动求导
multi_loss.backward()

# 查看 w 和 b 的梯度
print(w.grad)
print(b.grad)

# 更新一下参数
multi_learning_rate = 0.001
w.data = w.data - multi_learning_rate * w.grad.data
b.data = b.data - multi_learning_rate * b.grad.data

# 更新一次之后的模型
y_pred = my_multi_linear(x_train)

# plt.plot(x_train.data.numpy()[:, 0], y_pred.data.numpy(), label='fitting curve', color='r')
# plt.plot(x_train.data.numpy()[:, 0], y_sample, label='real curve', color='b')
# plt.legend()
# plt.show()

# 进行 200 次参数的迭代更新
epoch = 200  # 迭代次数

for e in range(epoch):
    y_pred = my_multi_linear(x_train)
    multi_loss = get_multi_loss(y_pred, y_train)

    # 将 w 和 b 的梯度先归零，这是因为之前我们已经更新过一次参数了，w 和 b 的梯度还存在值，如果前面
    # 我们没有进行求一次梯度，那么我们就需要加上判断是不是第一次梯度清零，第一次的话，系统会将 grad 初始化为 None，就会出错了
    #     if e != 0:
    w.grad.data.zero_()
    b.grad.data.zero_()
    multi_loss.backward()

    # 更新参数
    w.data = w.data - 0.01 * w.grad.data
    b.data = b.data - 0.01 * b.grad.data
    if (e + 1) % 20 == 0:
        print('Epoch [{}/{}], Multi_Loss: {:.4f}'.format(e + 1, epoch, multi_loss.data))

# 更新完成之后，将得到的结果画出来
y_pred = my_multi_linear(x_train)

# plt.plot(x_train.data.numpy()[:, 0], y_pred.data.numpy(), label='fitting curve', color='r')
# plt.plot(x_train.data.numpy()[:, 0], y_sample, label='real curve', color='b')
# plt.legend()
# plt.show()


# ------------------- 3 - sklearn 版 Linear Regression 实现 ---------------------
# ===== 3.1 - 简单版本 ====
# 导入相关库
from sklearn import datasets, linear_model  # 引入sklearn自带数据集，线性模型
from sklearn.metrics import mean_squared_error, r2_score  # 引入 sklearn 自带评价指标

# 加载 diabetes 数据集
diabetes_x, diabetes_y = datasets.load_diabetes(return_X_y=True)

# 我们只需要使用一个 feature（特征）
diabetes_x = diabetes_x[:, np.newaxis, 2]

# 将数据集分为 训练数据集（training） 和 测试数据集（testing）
diabetes_x_train = diabetes_x[:-20]  # 除了最后的 20 条，全部都作为训练集
diabetes_x_test = diabetes_x[-20:]  # 最后的 20 条，作为测试集

# 将目标（标签）分为 训练集 和 测试集
diabetes_y_train = diabetes_y[:-20]
diabetes_y_test = diabetes_y[-20:]

# # 将训练数据画在图上
# plt.plot(diabetes_x_train, diabetes_y_train, 'bo')
# plt.show()

# 创建 线性回归模型 对象
regr = linear_model.LinearRegression()

# 使用 fit 函数来对模型进行训练
regr.fit(diabetes_x_train, diabetes_y_train)

# 使用训练完成的模型 对 测试数据集 来作出预测（使用 predict 函数）
diabetes_y_pred = regr.predict(diabetes_x_test)

# # 将预测结果展示出来
# plt.plot(diabetes_x_train, diabetes_y_train, 'bo', label='Original data')
# plt.plot(diabetes_x_test, diabetes_y_pred, 'ro-', label='Fitted line')
# plt.legend()
# plt.show()

# ======= 3.2 -  多项式复杂版本 =========

# 创建简单多项式回归对象
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

model = Pipeline([('poly', PolynomialFeatures(degree=3)),
                  ('linear', LinearRegression(fit_intercept=False))])
# 拟合 3 阶多项式回归
x = np.arange(5)
y = 3 - 2 * x + x ** 2 - x ** 3
model = model.fit(x[:, np.newaxis], y)

# coef_ 参数代表的就是我们需要求得的 权重参数 w
model.named_steps['linear'].coef_
