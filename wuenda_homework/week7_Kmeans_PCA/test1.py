import numpy as np
import scipy.io as scio
import sys
import matplotlib.pyplot as plt
import random


# 用于导入数据的函数
def input_data():
    # 导入训练集的路径
    data_file = 'data/ex7data2.mat'
    # 导入训练集
    data = scio.loadmat(data_file)
    X = data['X']
    return X


# 用于打印原始数据的函数
def plot_origin_data(X):
    fig, ax = plt.subplots(1, 1)
    ax.scatter(X[:, 0], X[:, 1])
    plt.show()


# 用于寻找距离最近的聚类中心
def find_closest_centroids(X, centroids):
    idx = []  # 初始化一个存放中心的列表
    for i in range(0, X.shape[0]):  # 遍历每一个样本
        dis = sys.maxsize  # 初始化一个最大值距离
        k = -1  # 初始化当前样本中心
        for j in range(0, centroids.shape[0]):  # 遍历每一个聚类中心
            if np.sum(np.power(X[i, :] - centroids[j, :], 2)) < dis:  # 如果距离这个聚类中心更近
                k = j  # 更新中心
                dis = np.sum(np.power(X[i, :] - centroids[j, :], 2))  # 修改最小距离
        idx.append(k + 1)  # 列表添加每一个样本的中心
    return idx


# 更新聚类中心
def compute_centroids(X, centroids, idx):
    for i in range(0, centroids.shape[0]):  # 遍历每一个聚类中心
        idxx = [k for k in range(0, len(idx)) if idx[k] == (i + 1)]  # 提取属于当前聚类中心的样本位置
        cen_temp = np.zeros((1, centroids.shape[1]))  # 初始化一个临时变量用来进行向量求和
        for j in idxx:
            cen_temp += X[j]  # 属于当前聚类中心的样本向量求和
        centroids[i] = cen_temp / len(idxx)  # 得到新的聚类中心
    return centroids


# 随机初始化聚类中心
def Kmeans_init_centroids(X, K):
    random_int = []  # 定义一个随机数数组
    for i in range(0, K):
        random_int.append(random.randint(1, X.shape[0] + 1))  # 生成K个随机数
    random_centroids = np.array([], dtype=float)  # 创建聚类中心
    random_centroids = random_centroids.reshape(0, X.shape[1])
    # 根据随机数选择对应的样本作为聚类中心
    for k in random_int:
        random_centroids = np.insert(random_centroids, random_centroids.shape[0], values=X[k], axis=0)
    return random_centroids


# 运行Kmeans算法
def runKmeans(X, centroids, iter, K):
    iter_centroids = np.array([], dtype=float)  # 记录聚类中心变化的数组
    iter_centroids = iter_centroids.reshape(K, 0)
    idx = None  # 记录样本的所属聚类中心
    for i in range(0, iter):
        idx = find_closest_centroids(X, centroids)  # 为每一个样本寻找最近的聚类中心
        centroids = compute_centroids(X, centroids, idx)  # 更新聚类中心
        # 记录聚类中心的变化
        iter_centroids = np.insert(iter_centroids, iter_centroids.shape[1], values=centroids[:, 0], axis=1)
        iter_centroids = np.insert(iter_centroids, iter_centroids.shape[1], values=centroids[:, 1], axis=1)
    fig, ax = plt.subplots(1, 1)
    # 绘制原始数据的散点图
    for i in range(0, len(idx)):
        c = idx[i]
        if c == 1:
            ax.scatter(X[i, 0], X[i, 1], c='r')
        elif c == 2:
            ax.scatter(X[i, 0], X[i, 1], c='b')
        elif c == 3:
            ax.scatter(X[i, 0], X[i, 1], c='g')
    # 绘制三个聚类中心的变化
    ax.plot(iter_centroids[0, 0:iter_centroids.shape[1]:2], iter_centroids[0, 1:iter_centroids.shape[1]:2], c='black',
            marker='o')
    ax.plot(iter_centroids[1, 0:iter_centroids.shape[1]:2], iter_centroids[1, 1:iter_centroids.shape[1]:2], c='black',
            marker='o')
    ax.plot(iter_centroids[2, 0:iter_centroids.shape[1]:2], iter_centroids[2, 1:iter_centroids.shape[1]:2], c='black',
            marker='o')
    plt.show()

if __name__ == '__main__':
    K = 3  # 定义聚类中心的个数
    X = input_data()  # 导入数据
    centroids = Kmeans_init_centroids(X, K)  # 随机初始化聚类中心
    runKmeans(X, centroids, 10, K)  # 运行KMeans
