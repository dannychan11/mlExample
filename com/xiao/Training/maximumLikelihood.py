# -*- coding:utf-8 -*-
'''
最大似然函数，对已经发生的事件，取同时发生的概率最大，连乘,最小二乘法加最大似然估计来解决线性回归问题
h
'''
import numpy as np
import random
import matplotlib.pylab as plb
import matplotlib as mpl

#设置图形中文乱码
mpl.rcParams['font.sans-serif']=[u'simHei']
mpl.rcParams['axes.unicode_minus']=False

random.seed(28)
# 阶数为 9 升维
order = 10

# 生成样本点
x = np.arange(-1.5, 1.5, 0.02)
y = [((a * a - 1) ** 3 + (a - 0.5) ** 2 + 3 * np.sin(2 * a)**2) for a in x]
# ax.plot(x,y,color='r',linestyle='-',marker='')

#创造误差
#x_a = [b1 * (random.randint(90, 120)) / 100 for b1 in x]
#y_a = [b2 * (random.randint(90, 120)) / 100 for b2 in y]
x_a = x.tolist()
y_a = y
plb.plot(x_a, y_a, 'mo',ms=3,zorder=5,label='测试样本点')#ms参数设置点的大小

# 曲线拟合
# 创建矩阵
# 初始化二维数组
array_x = [[0 for i in range(order + 1)] for i in range(len(x_a))]
# 对二维数组赋值
for i in range(order + 1):
    for j in range(len(x_a)):
        array_x[j][i] = x_a[j] ** i

# 将赋值后的二维数组转化为矩阵
matx = np.mat(array_x)
matrix_A = matx.T * matx
yy = np.mat(y_a)
print(matx.shape)
print(yy.shape)
matrix_B = matx.T * yy.T
matAA = (matrix_A.I*matrix_B).tolist()

print(matAA)
#matAA = np.linalg.solve(matrix_A, matrix_B).tolist()

# 画出拟合后的曲线 θ1X1+θ2X2+...+θnXn xn是关于X的n次方
xxa = np.arange(-1.5, 1.5, 0.01)
yya = []
#代表有多少个对应的y值
for i in range(len(xxa)):
    yyy = 0.0
    #对应x的阶数
    for j in range(order + 1):
        dy = 1.0
        #取得对应的x值
        for k in range(j):
            dy *= xxa[i]
        # 取得对应x的y值
        dy *= matAA[j][0]
        #把他连加下就是真是的Y值
        yyy += dy
    yya.append(yyy)
plb.plot(xxa, yya, 'g-',label='预估函数',zorder=10)

plb.legend(loc='lower right')
plb.show()

