import numpy as np
import pandas as pd
import scipy.io as scio
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from scipy.optimize import minimize


# 用于导入数据的函数
def input_data():
    # 导入训练集的路径
    data_file = 'data/ex6data3.mat'
    # 导入训练集
    data = scio.loadmat(data_file)
    X = data['X']
    y = data['y']
    # 导入交叉验证集
    X_val = data['Xval']
    y_val = data['yval']
    return X, y, X_val, y_val

# 用于自动选择参数的函数
def auto_select_para(X, y, X_val, y_val):
    para_array = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 90]  # 定义参数数组
    accuary = 0  # 定义精确度
    opt_gamma = 0  # 定义最优gamma，gamma=1/（2*σ）平方，因此gamma越大，σ越小，偏差越小，方差越大
    opt_C = 0  # 定义最优C
    for p_gamma in para_array:  # 遍历gamma
        for p_C in para_array:  # 遍历C
            svc = SVC(C=p_C, kernel='rbf', gamma=p_gamma)
            svc.fit(X, y.flatten())  # 进行训练
            if svc.score(X_val, y_val) > accuary:  # 比较交叉验证集精确度
                accuary = svc.score(X_val, y_val)
                opt_gamma = p_gamma  # 进行替换
                opt_C = p_C  # 进行替换
    print(accuary)  # 打印精确度
    print(opt_C)  # 打印最佳C
    print(opt_gamma)  # 打印最佳gamma
    return opt_gamma, opt_C


# 用于可视化数据和绘制决策边界的函数
def plot_data_and_decision_bounday(X, y, opt_gamma, opt_C):
    # 绘制训练集数据的散点图
    fig, ax = plt.subplots(1, 1)
    for i in range(0, X.shape[0]):
        if y[i][0] == 1:
            ax.scatter(X[i][0], X[i][1], color='r', marker='+')
        else:
            ax.scatter(X[i][0], X[i][1], color='b', marker='o')
    ax.set_xticks(np.arange(-0.6, 0.4, 0.1))
    ax.set_yticks(np.arange(-0.8, 0.7, 0.2))

    # SVM分类器的训练
    svc = SVC(C=opt_C, kernel='rbf', gamma=opt_gamma)  # 构建一个线性分类器，参数采用最优值
    svc.fit(X, y.flatten())  # 使用（X，y）来进行训练，注意这里的y要转换成一维向量
    x_min, x_max = -0.6, 0.4  # 定义x最小最大
    y_min, y_max = -0.8, 0.7  # 定义y最小最大
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500),  # 生成网格
                         np.linspace(y_min, y_max, 500))

    z = svc.predict(np.c_[xx.flatten(), yy.flatten()])  # 通过训练好的分类器进行预测
    zz = z.reshape(xx.shape)  # 重构一下维度
    ax.contour(xx, yy, zz)  # 绘制等高线图
    plt.show()


if __name__ == '__main__':
    X, y, X_val, y_val = input_data()  # 导入数据
    opt_gamma, opt_C = auto_select_para(X, y, X_val, y_val)  # 自动选择最优参数
    plot_data_and_decision_bounday(X, y, opt_gamma, opt_C)  # 绘制决策边界
