import numpy as np
import scipy.io as scio
import math
import matplotlib.pyplot as plt


# 用于导入数据的函数
def input_data():
    # 导入训练集的路径
    data_file = 'data/ex8data1.mat'
    # 导入训练集
    data = scio.loadmat(data_file)
    X = data['X']
    X_val = data['Xval']
    y_val = data['yval']
    # # 可视化原始数据
    # fig,ax = plt.subplots(1,1)
    # ax.scatter(X[:,0],X[:,1])
    # plt.show()
    return X, X_val, y_val


# 用于计算单变量高斯分布
def univariate_gaussian_distribution(X, mean, var):
    # 根据单变量高斯分布公式计算每种特征的概率
    p = np.exp(-np.power((X - mean), 2) / (2 * var)) / (np.sqrt(2 * math.pi * var))
    answer_p = p[:, 0]
    for i in range(1, p.shape[1]):
        answer_p = answer_p * p[:, i]  # 假设相互独立，累乘计算总体概率
    return answer_p


# 用于计算多变量高斯分布
def multivariate_gaussian_distribution(X, mean, covariance):
    # 根据多变量高斯分布公式计算总的概率
    p = np.exp(-0.5 * np.diag((X - mean) @ np.linalg.inv(covariance) @ (X - mean).T)) / (
            np.power((2 * math.pi), 0.5 * X.shape[1]) * np.power(np.linalg.det(covariance), 0.5))
    return p


# 用于选取阈值的函数
def select_threshold(X_val, y_val, mean, variance, covariance):
    p = univariate_gaussian_distribution(X_val, mean, variance)  # 调用单变量高斯分布函数
    # p = multivariate_gaussian_distribution(X_val, mean, covariance)  # 调用多变量高斯分布函数
    choose_epsilon = 0  # 定义初始阈值
    f = 0  # 定义初始f值
    for epsilon in np.linspace(min(p), max(p), 1000):  # 遍历每一种阈值
        tp = 0  # true positive
        fp = 0  # false positive
        fn = 0  # false negative
        for i in range(0, p.shape[0]):  # 遍历每一个概率
            if p[i] < epsilon and y_val[i] == 1:  # true positive
                tp += 1
            elif p[i] < epsilon and y_val[i] == 0:  # false positive
                fp += 1
            elif p[i] >= epsilon and y_val[i] == 1:  # false negative
                fn += 1
        prec = tp / (tp + fp) if (tp + fp) != 0 else 0  # 计算精确率
        rec = tp / (tp + fn) if (tp + fn) != 0 else 0  # 计算召回率
        com_f = 2 * prec * rec / (prec + rec) if (prec + rec) != 0 else 0  # 计算f值
        if com_f > f:
            choose_epsilon = epsilon  # 如果得到更好f值的阈值，那么就进行更新
            f = com_f
    return choose_epsilon, f


# 用于绘制等高线的函数
def plotContour(X, X_val, y_val, mean, variance, covariance):
    choose_epsilon, f = select_threshold(X_val, y_val, mean, variance, covariance)  # 调用选取最优阈值的函数
    p = univariate_gaussian_distribution(X, mean, variance)  # 调用单变量高斯分布计算概率
    # p = multivariate_gaussian_distribution(X, mean, covariance)  # 调用多变量高斯分布计算概率
    # 可视化原始数据
    fig, ax = plt.subplots(1, 1)
    ax.scatter(X[:, 0], X[:, 1])  # 绘制原始数据散点
    anoms = np.array([X[i] for i in range(0, X.shape[0]) if p[i] < choose_epsilon])  # 挑选异常点
    ax.scatter(anoms[:, 0], anoms[:, 1], color='r', marker='x')  # 标记异常点
    # 用于绘制等高线
    x_min, x_max = 0, 30
    y_min, y_max = 0, 30
    x = np.arange(x_min, x_max, 0.3)
    y = np.arange(y_min, y_max, 0.3)
    xx, yy = np.meshgrid(x, y)  # 生成网格
    z = univariate_gaussian_distribution(np.c_[xx.ravel(), yy.ravel()], mean, variance)  # 调用单变量高斯分布
    # z = multivariate_gaussian_distribution(np.c_[xx.ravel(), yy.ravel()], mean, covariance)  # 调用多变量高斯分布
    zz = z.reshape(xx.shape)  # 重构一下维度
    cont_levels = [10 ** h for h in range(-20, 0, 3)]
    ax.contour(xx, yy, zz, cont_levels)  # 绘制等高线图
    plt.show()


if __name__ == '__main__':
    X, X_val, y_val = input_data()  # 导入数据
    # 特别注意：在绘制等高线时要使用原始数据的均值、方差、协方差
    mean = np.mean(X, axis=0)  # 计算原始数据的均值
    variance = np.var(X, axis=0)  # 计算方差，用于单变量高斯分布
    covariance = np.cov(X, rowvar=False)  # 计算协方差，用于多变量高斯分布
    choose_epsilon, f = select_threshold(X_val, y_val, mean, variance, covariance)  # 挑选阈值
    print('choose_epsilon:', choose_epsilon, ' f:', f)  # 打印阈值和f
    plotContour(X, X_val, y_val, mean, variance, covariance)  # 绘制等高线图
