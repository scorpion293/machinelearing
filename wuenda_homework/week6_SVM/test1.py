import numpy as np
import scipy.io as scio
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC


# 用于导入数据的函数
def input_data():
    # 导入训练集的路径
    data_file = 'data/ex6data1.mat'
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
    # 设置坐标轴
    plt.xticks(np.arange(0, 5, 0.5))
    plt.yticks(np.arange(1.5, 5.5, 0.5))

    # SVM分类器的训练
    # C越小，正则化强度越大，越容易欠拟合
    svc = SVC(C=100, kernel='linear')  # 构建一个线性分类器，其中C=100
    svc.fit(X, y.flatten())  # 使用（X，y）来进行训练，注意这里的y要转换成一维向量
    x_min, x_max = -0.5, 4.5  # 定义x最小最大
    y_min, y_max = 1.5, 5  # 定义y最小最大
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500),  # 生成网格
                         np.linspace(y_min, y_max, 500))
    # np.c_按列🔗矩阵，columns
    # 形成一个大小为25w个点的测试数据组
    z = svc.predict(np.c_[xx.flatten(), yy.flatten()])  # 通过训练好的分类器进行预测
    zz = z.reshape(xx.shape)  # 重构一下维度
    plt.contour(xx, yy, zz)  # 绘制等高线图
    plt.show()


if __name__ == '__main__':
    X, y = input_data()  # 导入数据
    plot_data_and_decision_bounday(X, y)  # 绘制决策边界
