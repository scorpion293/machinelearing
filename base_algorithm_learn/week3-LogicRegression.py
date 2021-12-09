import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, plot_confusion_matrix
from matplotlib.colors import ListedColormap


def two_classification():
    # 读取数据
    df = pd.read_csv('data/hearing_test.csv')

    # seaborn是一个更强大的绘图库，scatterplot可以根据属性的不同绘制出颜色不同、大小不同、形状不同的点
    sns.scatterplot(x='age', y='physical_score', data=df, hue='test_result')
    # pairplot可以显示变量两者之间的关系
    sns.pairplot(df, hue='test_result')

    # 准备数据
    # df.drop是删除列或或者行，要写明axis=0或1
    X = df.drop('test_result', axis=1)
    y = df['test_result']
    # train_test_split用来划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=50)
    # StandardScaler用来进行数据预处理
    scaler = StandardScaler()
    scaled_X_train = scaler.fit_transform(X_train)
    scaled_X_test = scaler.transform(X_test)

    # 定义模型
    log_model = LogisticRegression()

    # 训练模型
    log_model.fit(scaled_X_train, y_train)

    # 预测数据
    y_pred = log_model.predict(scaled_X_test)
    # 计算预测准确率
    accuracy = accuracy_score(y_test, y_pred)
    print(accuracy)
    # 绘制混淆矩阵
    plot_confusion_matrix(log_model, scaled_X_test, y_test)
    plt.show()
    # 输出precision，recall，accuracy
    print(classification_report(y_test, y_pred))


def mul_classification():
    df = pd.read_csv('data//iris.csv')
    sns.scatterplot(x='sepal_length', y='sepal_width', data=df, hue='species')
    plt.show()
    sns.scatterplot(x='petal_length', y='petal_width', data=df, hue='species')
    plt.show()
    sns.pairplot(df,hue='species')
    plt.show()

    # 准备数据
    X = df.drop(columns=['species'])
    y = df['species']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=50)
    scaler = StandardScaler()
    scaled_X_train = scaler.fit_transform(X_train)
    scaled_X_test = scaler.transform(X_test)

    # 定义模型
    # lbfgs一般用于较小的数据集，saga一般用于较大的数据集
    # C值越小，正则化强度越大，默认为1，一般可以从0.01开始设置，每次翻10倍，直到1000
    softmax_model = LogisticRegression(multi_class="multinomial", solver="lbfgs", C=1, random_state=50)

    # 训练模型
    softmax_model.fit(scaled_X_train, y_train)

    # 预测数据
    y_pred = softmax_model.predict(scaled_X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(accuracy)
    print(classification_report(y_test, y_pred))

def mul_classification_by_two():
    df = pd.read_csv('data//iris.csv')
    # 提取特征
    # 将df数据转换为numpy数组
    X = df[['petal_length', 'petal_width']].to_numpy()
    # 将多个形式的分类结果转换为数字
    y = df["species"].factorize(['setosa', 'versicolor', 'virginica'])[0]
    # 定义模型
    softmax_reg = LogisticRegression(multi_class="multinomial", solver="lbfgs", C=1, random_state=50)

    # 训练模型
    softmax_reg.fit(X, y)

    # 随机测试数据
    # meshgrid将x按行往下展开，将y按列往右展开
    x0, x1 = np.meshgrid(
        np.linspace(0, 8, 500).reshape(-1, 1),
        np.linspace(0, 3.5, 200).reshape(-1, 1),
    )
    # np.r_：是按列连接两个矩阵，就是把两矩阵上下相加，要求列数相等，类似于pandas中的concat()
    # np.c_：是按行连接两个矩阵，就是把两矩阵左右相加，要求行数相等，类似于pandas中的merge()
    X_new = np.c_[x0.ravel(), x1.ravel()]

    # 预测
    y_proba = softmax_reg.predict_proba(X_new)
    y_predict = softmax_reg.predict(X_new)

    # 绘制图像
    zz1 = y_proba[:, 1].reshape(x0.shape)
    zz = y_predict.reshape(x0.shape)
    plt.figure(figsize=(10, 4))
    plt.plot(X[y == 2, 0], X[y == 2, 1], "g^", label="Iris virginica")
    plt.plot(X[y == 1, 0], X[y == 1, 1], "bs", label="Iris versicolor")
    plt.plot(X[y == 0, 0], X[y == 0, 1], "yo", label="Iris setosa")

    custom_cmap = ListedColormap(['#fafab0', '#9898ff', '#a0faa0'])
    plt.contourf(x0, x1, zz, cmap=custom_cmap)
    contour = plt.contour(x0, x1, zz1, cmap=plt.cm.brg)
    plt.clabel(contour, inline=1, fontsize=12)
    plt.xlabel("Petal length", fontsize=14)
    plt.ylabel("Petal width", fontsize=14)
    plt.legend(loc="center left", fontsize=14)
    plt.axis([0, 7, 0, 3.5])
    plt.show()



if __name__ == '__main__':
    mul_classification_by_two()
