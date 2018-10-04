# -- encoding:utf-8 --
"""
功能：降低代码的编程量
Create by ibf on 2018/6/23
"""
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.externals import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# 设置字符集，防止中文乱码
mpl.rcParams['font.sans-serif'] = [u'simHei']
mpl.rcParams['axes.unicode_minus'] = False

# 1. 加载数据
path = './datas/household_power_consumption_201.txt'
df = pd.read_csv(filepath_or_buffer=path, sep=';')
# print(df.info())

# 2. 数据的清洗
# inplace: 当设置为True的时候，表示对原始的DataFrame做修改；默认为False
df.replace('?', np.nan, inplace=True)
# DataFrame: axis=0表示对行做处理，axis=1表示对列做处理
# how: 可选参数any和all，any表示只要有任意一个为nan，那么进行数据删除；如果为all，表示所有值为nan的时候数据删除
# 功能：只要有任意一个样本中的任意数据特征属性为nan的形式，就就将该样本删除。
df = df.dropna(axis=0, how='any')
# print(df.info())

# 3. 模型所需要的特征属性获取(构建特征属性矩阵X和目标属性Y)
X = df.iloc[:, 2:4]
Y = df.iloc[:, 5]

# 4. 数据分割
x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size=0.7, random_state=28)

# 5. 特征工程
# 操作可选
# 主要就是：哑编码、连续数据离散化、数据的转换、标准化、归一化、降维.....
ss = StandardScaler()

# 6. 算法/模型对象构建
algo = LinearRegression()

# 6.2 构建管道对象
pipeline = Pipeline([
    ('ss_name', ss),  # 给定第一步的名称以及对应的操作对象
    ('algo_name', algo)  # 给定第二步的名称以及对应的操作对象
])

# 获取管道中对应的步骤（根据你给定的名称来获取）
ss2 = pipeline.get_params()['ss_name']
print(type(ss2))

# 设置参数的时候根据设置的步骤名称以及对应步骤对象的属性值来设置参数
pipeline.set_params(algo_name__fit_intercept=False)
print(pipeline.get_params()['algo_name'])

# 7. 算法模型训练
pipeline.fit(x_train, y_train)

# 8. 模型效果评估
# 可选，主要目的就是评估一下模型的效果
# 主要就是根据算法类型，分别选择精确率、准确率、F1、RMSE、MSE、R2等指标进行校验
print("模型效果：{}".format(pipeline.score(x_train, y_train)))

# 9. 模型的保存
"""
两种保存方式：
1. 直接保存模型对象
2. 保存模型的预测结果到数据库或者其它数据存储的位置
"""
# a. 直接保存模型对象
joblib.dump(pipeline, './model/pipline')

# b. 保存预测结果
# 获取预测结果
y_hat = pipeline.predict(x_test)
print(y_hat)

# 画图看一下效果如何
t = np.arange(len(x_test))
plt.figure(facecolor='w')
plt.plot(t, y_test, 'r-', linewidth=2, label=u'真实值')
plt.plot(t, y_hat, 'g-', linewidth=2, label=u'预测值')
plt.legend(loc='lower right')
plt.title('线性回归')
plt.show()
