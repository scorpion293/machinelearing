import numpy as np
import pandas as pd
import scipy.io as scio
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from scipy.optimize import minimize


# 用于导入数据的函数
def input_data():
    # 导入训练集的路径
    data_file = 'data/ex6data2.mat'
    # 导入训练集
    data = scio.loadmat(data_file)
    X = data['X']
    y = data['y']
    return X, y


# 用于可视化数据和绘制决策边界的函数
def plot_data_and_decision_bounday(X, y):
    # 绘制训练集数据的散点图
    for i in range(0, X.shape[0]):
        if y[i][0] == 1:
            plt.scatter(X[i][0], X[i][1], color='r', marker='+')
        else:
            plt.scatter(X[i][0], X[i][1], color='b', marker='o')
    plt.xticks(np.arange(0, 1.1, 0.1))
    plt.yticks(np.arange(0.4, 1.1, 0.1))

    # SVM分类器的训练
    # 构建一个线性分类器，这里尝试修改gamma，gamma=1/（2*σ）平方，因此gamma越大，σ越小，偏差越小，方差越大
    svc = SVC(C=100, kernel='rbf', gamma=100)
    svc.fit(X, y.flatten())  # 使用（X，y）来进行训练，注意这里的y要转换成一维向量
    x_min, x_max = -0, 1.1  # 定义x最小最大
    y_min, y_max = 0.4, 1.1  # 定义y最小最大
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500),
                         np.linspace(y_min, y_max, 500))

    z = svc.predict(np.c_[xx.flatten(), yy.flatten()])  # 生成网格
    zz = z.reshape(xx.shape)  # 重构一下维度
    plt.contour(xx, yy, zz)  # 绘制等高线图
    plt.show()


if __name__ == '__main__':
    X, y = input_data()  # 导入数据
    plot_data_and_decision_bounday(X, y)  # 绘制决策边界
