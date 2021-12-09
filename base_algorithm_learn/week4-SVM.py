import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

# 绘制图像
# 定义绘制SVM边界方法
def plot_svm_boundary(model, X, y):
    # values将df转换为列表数组
    X = X.values
    y = y.values

    # 根据数组y的值来绘制不同颜色的点，y=[0,0,0,1,0,1,1....]，cmap将这些数字映射为对应的颜色
    plt.scatter(X[:, 0], X[:, 1], c=y, s=30, cmap='coolwarm')

    # 获取当前图标
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # 在x，y的范围内各自产生30个值，然后两两匹配，其实和用两个for循环效果差不多
    # 其实和以下方法没区别,但还是推荐用这样方式，np.meshgrid这个函数挺常用的
    # xx = np.linspace(0, 10, 30)
    # yy = np.linspace(10, 20, 20)
    # xy = []
    # for i in range(xx.shape[0]):
    #     for j in range(yy.shape[0]):
    #         xy.append([xx[i], yy[j]])
    # xy = np.array(xy)

    xx = np.linspace(xlim[0], xlim[1], 30)
    yy = np.linspace(ylim[0], ylim[1], 30)
    XX, YY= np.meshgrid(xx, yy)
    # np.vstack将两个矩阵沿竖直方向堆叠
    xy = np.vstack([XX.flatten(), YY.flatten()]).T
    # decision_function：输出样本距离各个分类器的分隔超平面的置信度，并由此可以推算出predict的预测结果
    # predict_procaba：输出样本属于各个类别的概率值，并由此可以推算出predict的预测结果
    # predict：输出样本属于具体类别的预测结果
    Z = model.decision_function(xy).reshape(XX.shape)

    # counter画等高线，counterf填充不同等高线部分的颜色
    # alpha是线的透明度,levels是距离svm分割线的距离
    ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,
               linestyles=['--', '-', '--'])
    # plot support vectors
    ax.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1], s=100,
               linewidth=1, facecolors='none', edgecolors='k')
    plt.show()


if __name__ == '__main__':
    df = pd.read_csv('data/mouse_viral_study.csv')
    print(df.describe())
    sns.scatterplot(x='Med_1_mL', y='Med_2_mL', hue='Virus Present', data=df)
    plt.show()
    sns.pairplot(df, hue='Virus Present')
    plt.show()

    # 准备数据
    y = df['Virus Present']
    X = df.drop(['Virus Present'], axis=1)

    # 定义模型
    # model = SVC(kernel='linear', C=1000)
    # model = SVC(kernel='linear', C=0.05)
    # model = SVC(kernel='poly', C=0.05, degree=5)
    model = SVC(kernel='poly', C=0.05,degree=5)

    # 训练模型
    model.fit(X, y)
    plot_svm_boundary(model, X, y)

    # svm提供的一个调参函数
    svm = SVC()
    param_grid = {'C': [0.01, 0.1, 1], 'kernel': ['rbf', 'poly', 'linear', 'sigmoid'], 'gamma': [0.01, 0.1, 1]}
    grid = GridSearchCV(svm, param_grid)
    grid.fit(X, y)
    print("grid.best_params_ = ", grid.best_params_, ", grid.best_score_ =", grid.best_score_)
