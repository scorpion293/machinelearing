from sklearn.linear_model import LinearRegression

'''
模型常用的函数
'''
# # 拟合模型
# model.fit(X_train, y_train)
# # 模型预测
# model.predict(X_test)
# # 获得这个模型的参数
# model.get_params()
# # 为模型进行打分
# model.score(data_X, data_y)  # 线性回归：R square； 分类问题： acc

from sklearn.linear_model import LinearRegression

# 定义线性回归模型
model = LinearRegression(fit_intercept=True, normalize=False,
                         copy_X=True, n_jobs=1)
"""
参数
    fit_intercept：是否计算截距。False-模型没有截距
    normalize： 当fit_intercept设置为False时，该参数将被忽略。 如果为真，则回归前的回归系数X将通过减去平均值并除以l2-范数而归一化。
    n_jobs：指定线程数
"""

from sklearn.linear_model import LogisticRegression

# 定义逻辑回归模型
model = LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=1.0,
                           fit_intercept=True, intercept_scaling=1, class_weight=None,
                           random_state=None, solver='liblinear', max_iter=100, multi_class='ovr',
                           verbose=0, warm_start=False, n_jobs=1)
"""参数
    penalty：使用指定正则化项（默认：l2）
    dual: n_samples > n_features取False（默认）
    C：正则化强度的反，值越小正则化强度越大
    n_jobs: 指定线程数
    random_state：随机数生成器
    fit_intercept: 是否需要常量
"""

from sklearn import naive_bayes

model = naive_bayes.GaussianNB()  # 高斯贝叶斯
model = naive_bayes.MultinomialNB(alpha=1.0, fit_prior=True, class_prior=None)
model = naive_bayes.BernoulliNB(alpha=1.0, binarize=0.0, fit_prior=True, class_prior=None)
"""
文本分类问题常用MultinomialNB
参数
    alpha：平滑参数
    fit_prior：是否要学习类的先验概率；false-使用统一的先验概率
    class_prior: 是否指定类的先验概率；若指定则不能根据参数调整
    binarize: 二值化的阈值，若为None，则假设输入由二进制向量组成
"""

from sklearn import tree

model = tree.DecisionTreeClassifier(criterion='gini', max_depth=None,
                                    min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0,
                                    max_features=None, random_state=None, max_leaf_nodes=None,
                                    min_impurity_decrease=0.0, min_impurity_split=None,
                                    class_weight=None, presort=False)
"""参数
    criterion ：特征选择准则gini/entropy
    max_depth：树的最大深度，None-尽量下分
    min_samples_split：分裂内部节点，所需要的最小样本树
    min_samples_leaf：叶子节点所需要的最小样本数
    max_features: 寻找最优分割点时的最大特征数
    max_leaf_nodes：优先增长到最大叶子节点数
    min_impurity_decrease：如果这种分离导致杂质的减少大于或等于这个值，则节点将被拆分。
"""

from sklearn.svm import SVC

model = SVC(C=1.0, kernel='rbf', gamma='auto')
"""参数
    C：误差项的惩罚参数C
    gamma: 核相关系数。浮点数，If gamma is ‘auto’ then 1/n_features will be used instead.
"""

from sklearn.neural_network import MLPClassifier

# 定义多层感知机分类算法
model = MLPClassifier(activation='relu', solver='adam', alpha=0.0001)
"""参数
    hidden_layer_sizes: 元祖
    activation：激活函数
    solver ：优化算法{‘lbfgs’, ‘sgd’, ‘adam’}
    alpha：L2惩罚(正则化项)参数。
"""
