from sklearn import preprocessing

def data_standardize():
    '''
    数据标准化
    数据标准化后会使得每个特征中的数值平均变为0（将每个特征的值都减掉原始资料中该特征的平均），标准差变为1
    这个方法被广泛的使用在许多机器学习算法中（例如：支持向量机，逻辑回归和类神经网络）
    '''
    train_data = [[-16, 0], [-4, 0], [15, 1], [30, 1]]
    test_data = [[-3, 0], [8, 1]]
    # 先创建scaler
    scaler = preprocessing.StandardScaler()
    # 再基于训练数据使用fit
    scaler.fit(train_data)
    # 最后对train和test数据使用transfer
    train_data = scaler.transform(train_data)
    test_data = scaler.transform(test_data)
    print(train_data)
    print(test_data)

def minmax_standardize():
    '''
    最小最大值标准化
    最小最大规范化对原始数据进行线性变换，变换到[0,1]区间（也可以是其他固定最小最大值的区间）
    '''
    train_data = [[-16, 0], [-4, 0], [15, 1], [30, 1]]
    test_data = [[-3, 0], [8, 1]]
    # 先创建scaler
    # feature_range: 定义归一化范围，注用（）括起来
    scaler=preprocessing.MinMaxScaler(feature_range=(0, 1))
    # 再基于训练数据使用fit
    scaler.fit(train_data)
    # 最后对train和test数据使用transfer
    train_data=scaler.transform(train_data)
    test_data=scaler.transform(test_data)
    print(train_data)
    print(test_data)

def data_normalize():
    '''
    数据正则化
    当你想要计算两个样本的相似度时必不可少的一个操作，就是正则化。
    其思想是：首先求出样本的p范数，然后该样本的所有元素都要除以该范数，这样最终使得每个样本的范数都是1。
    规范化（Normalization）是将不同变化范围的值映射到相同的固定范围，常见的是[0,1]，也成为归一化
    '''
    X = [[1., -1., 2.],[2., 0., 0.],[0., 1., -1.]]
    X_normalized = preprocessing.normalize(X, norm='l2')
    print(X_normalized)

def onehot_code():
    '''
    one-hot编码是一种对离散特征值的编码方式，在LR模型中常用到，用于给线性模型增加非线性能力
    '''
    data = [[0, 0, 3], [1, 1, 0], [0, 2, 1], [1, 0, 2]]
    encoder = preprocessing.OneHotEncoder()
    encoder.fit(data)
    data=encoder.transform(data).toarray()
    print(data)

def data_binarization():
    '''
    给定阈值，将特征转换为0/1
    '''
    binarizer = preprocessing.Binarizer(threshold=1.1)
    X = [[1., -1., 2.], [2., 0., 0.], [0., 1., -1.]]
    X=binarizer.transform(X)

def label_encode():
    '''
    标签编码
    '''
    le = preprocessing.LabelEncoder()
    le.fit([1, 2, 2, 6])
    print(le.transform([1, 1, 2, 6]) ) #array([0, 0, 1, 2])
    #非数值型转化为数值型
    le.fit(["paris", "paris", "tokyo", "amsterdam"])
    print(le.transform(["tokyo", "tokyo", "paris"]))#array([2, 2, 1])




if __name__ == '__main__':
    label_encode()