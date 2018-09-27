# -- encoding:utf-8 --
"""
Create by ibf on 2018/6/24
"""

import pandas as pd
import numpy as np
import matplotlib as mpl

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# 设置字符集，防止中文乱码
mpl.rcParams['font.sans-serif'] = [u'simHei']
mpl.rcParams['axes.unicode_minus'] = False

# 1. 加载数据
names = ['id', 'Clump Thickness', 'Uniformity of Cell Size', 'Uniformity of Cell Shape',
         'Marginal Adhesion', 'Single Epithelial Cell Size', 'Bare Nuclei',
         'Bland Chromatin', 'Normal Nucleoli', 'Mitoses', 'Class']
path = '../datas/breast-cancer-wisconsin.data'
df = pd.read_csv(filepath_or_buffer=path, sep=',', header=None, names=names)
# print(df.info())

# 2. 数据清洗
df = df.replace('?', np.nan)
df = df.dropna(axis=0, how='any')
# print(df.info())

# 3. 数据获取
X = df[names[1:10]]
Y = df[names[10]]
print(Y.value_counts())

# 4. 数据的划分
x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size=0.9, random_state=28)
print(x_train.shape)
print(x_test.shape)

# 5. 特征工程
# 做一下标准化操作
ss = StandardScaler()
# # 模型训练
# ss.fit(x_train)
# # 对数据做一个转换操作
# x_train = ss.transform(x_train)
# 直接做模型训练 + 数据转换操作
x_train = ss.fit_transform(x_train)

# 6. 模型算法构建
"""
penalty='l2', 采用什么正则化，默认L2正则
dual=False, 
tol=1e-4, 模型训练过程中，算法模型对应score API返回值在两次迭代中的变换大小小于该值的时候，模型结束训练
C=1.0, 正则项/惩罚项的系数
fit_intercept=True, 在做线性转换的时候是否训练截距项，True就表示训练
intercept_scaling=1, 
class_weight=None, 给定每个类别的权重
random_state=None, 随机数种子
solver='liblinear', 模型的求解方式
max_iter=100, 给定模型最多的数据迭代次数
multi_class='ovr', 给定模型在对于多分类的时候采用什么方式进行模型的构建，该参数在二分类的时候没有效果。
verbose=0, warm_start=False, n_jobs=1
"""
algo = LogisticRegression(penalty='l2', fit_intercept=False, C=0.1, max_iter=10, tol=1e-4)

# 算法模型训练
algo.fit(x_train, y_train)

# 模型效果输出
print("训练集上模型效果/准确率：{}".format(algo.score(x_train, y_train)))
print("测试集上模型效果/准确率：{}".format(algo.score(x_test, y_test)))

# 获取预测值
print("实际值:")
print(y_test.ravel())
y_pred = algo.predict(x_test)
print("*" * 50)
print("预测值：(直接返回所属类别)")
print(y_pred)
print("*" * 50)
print("预测值：(返回所属类别的概率)")
print(algo.predict_proba(x_test))
print("决策函数值:(也就是线性转换的值θx)")
print(algo.decision_function(x_test))
print("预测值：(返回所属类别的概率的log转换值)")
print(algo.predict_log_proba(x_test))
