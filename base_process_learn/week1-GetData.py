from sklearn import datasets
import matplotlib.pyplot as plt


def import_dataset():
    '''
    导入数据集
    :return:
    '''
    # 导入鸢尾花数据集
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target

    # 导入手写数据集
    digits = datasets.load_digits()
    print(digits.data.shape)
    print(digits.target.shape)
    print(digits.images.shape)
    # 显示手写数字图片
    plt.matshow(digits.images[0])
    plt.show()


def create_dataset():
    '''
    生成数据集
    :return:
    '''
    # 生成用于分类的数据
    # n_samples：指定样本数
    # n_features：指定特征数
    # n_classes：指定几分类
    # random_state：随机种子，使得随机状可重
    X, y = datasets.make_classification(n_samples=6, n_features=5, n_classes=2, random_state=20)
    print(X)
    print(y)

    # 生成用于聚类的数据
    # n_samples表示产生多少个数据
    # n_features表示数据是几维
    # centers表示数据点中心，可以输入int数字，代表有多少个中心，也可以输入几个坐标
    # cluster_std表示分布的标准差
    data, target = datasets.make_blobs(n_samples=100, n_features=2, centers=5)
    plt.scatter(data[:, 0], data[:, 1], c=target)
    plt.show()
    # 为每个类别设置不同的方差
    data,target = datasets.make_blobs(n_samples=100,n_features=2,centers=3,cluster_std=[1.0,2.0,3.0])
    plt.scatter(data[:, 0], data[:, 1], c=target)
    plt.show()



if __name__ == '__main__':
    create_dataset()
