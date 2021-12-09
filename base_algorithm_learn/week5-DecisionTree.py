import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import AdaBoostClassifier


def decision_tree():
    # 引入数据集
    df = pd.read_csv('data/iris.csv')

    # 决策树模型
    X = df[['petal_length', 'petal_width']].to_numpy()
    y = df['species']
    tree_clf = DecisionTreeClassifier(max_depth=2, random_state=42)
    tree_clf.fit(X, y)
    plt.figure(figsize=(12, 8))
    plot_tree(tree_clf, filled=True)
    plt.show()

def random_forest():
    df = pd.read_csv("data/penguins_size.csv")
    df = df.dropna()
    # sns.pairplot(df, hue='species')
    # plt.show()
    X = pd.get_dummies(df.drop('species', axis=1), drop_first=True)
    y = df['species']
    # pd.set_option('display.max_columns', None)
    # print(X.head())
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
    # n_rstimators相当于C，太小容易欠拟合，太大容易过拟合
    model = RandomForestClassifier(n_estimators=10, max_features='auto', random_state=101)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    print(preds)
    accuracy_score(y_test, preds)

def auto_adjust_par():
    df = pd.read_csv("data/penguins_size.csv")
    df = df.dropna()
    X = pd.get_dummies(df.drop('species', axis=1), drop_first=True)
    y = df['species']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
    ada_clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), random_state=101)
    ada_clf.fit(X_train, y_train)
    param_grid = {'n_estimators': [10, 15, 20, 25, 30, 35, 40], 'learning_rate': [0.01, 0.1, 0.5, 1],
                  'algorithm': ['SAMME', 'SAMME.R']}
    grid = GridSearchCV(ada_clf, param_grid)
    grid.fit(X_train, y_train)
    print("grid.best_params_ = ", grid.best_params_, ", grid.best_score_ =", grid.best_score_)


if __name__ == '__main__':
    auto_adjust_par()


