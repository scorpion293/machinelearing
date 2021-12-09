import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# 初始化数据，绘制图像
celsius = [[-40], [-10], [0], [8], [15], [22], [38]]
fahrenheit = [[-40], [14], [32], [46.4], [59], [71.6], [100.4]]
plt.scatter(celsius, fahrenheit, color='red', label='real')
plt.xlabel('celsius')
plt.ylabel('fahrenheit')
plt.legend()
plt.grid(True)
plt.title('real data')
plt.show()

# 构建线性回归模型，训练数据
lr = LinearRegression()
# 注意线性模型只能接受2维数组，要用reshape(-1,1)
lr.fit(celsius,fahrenheit)

# 用训练好的模型预测测试数据
celsius_test = [[-50],[-30],[10],[20],[50],[70]]
fahrenheit_test = lr.predict(celsius_test)
plt.scatter(celsius,fahrenheit, color='red', label='real')
plt.scatter(celsius_test,fahrenheit_test, color='orange', label='predicted')
plt.xlabel('celsius')
plt.ylabel('fahrenheit')
plt.legend()
plt.grid(True)
plt.title('estimated vs real data')
plt.show()
