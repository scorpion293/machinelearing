import scipy.io as scio
from sklearn.svm import SVC


# 用于导入数据的函数
def input_data():
    # 导入训练集的路径
    data_file1 = 'data/spamTrain.mat'
    data_file2 = 'data/spamTest.mat'
    data1 = scio.loadmat(data_file1)
    data2 = scio.loadmat(data_file2)
    # 导入训练集
    X = data1['X']
    y = data1['y']
    # 导入测试集
    X_test = data2['Xtest']
    y_test = data2['ytest']
    return X, y, X_test, y_test


# 自动选择参数
def auto_select_para(X, y, X_val, y_val):
    para_array = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]  # 定义参数数组
    accuary = 0  # 定义精确度
    opt_gamma = 0  # 定义最优gamma，gamma=1/（2*σ）平方，因此gamma越大，σ越小，偏差越小，方差越大
    opt_C = 0  # 定义最优C
    for p_gamma in para_array:  # 遍历gamma
        for p_C in para_array:  # 遍历C
            svc = SVC(C=p_C, kernel='linear', gamma=p_gamma)  # 构建一个线性分类器
            svc.fit(X, y.flatten())  # 进行训练
            if svc.score(X_val, y_val) > accuary:  # 比较交叉验证集精确度
                accuary = svc.score(X_val, y_val)
                opt_gamma = p_gamma  # 进行替换
                opt_C = p_C  # 进行替换
    print('accuary',accuary*100)  # 打印精确度


if __name__ == '__main__':
    X, y, X_test, y_test = input_data()  # 导入数据
    auto_select_para(X, y, X_test, y_test)  # 自动选择最优参数并计算精确度
