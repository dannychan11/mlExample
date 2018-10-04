# -*- coding:utf-8 -*-
'''
方差、标准差等等
'''

import matplotlib.pylab as plb
import numpy as np
import matplotlib as mpl

#设置图形中文乱码
mpl.rcParams['font.sans-serif']=[u'simHei']
mpl.rcParams['axes.unicode_minus']=False

#指定随机变量算子
np.random.seed(10)
#x = np.random.randint(100,size=10)
x = np.random.rand(10)
y = np.random.randint(10,size=10)
print('打印变量%s'%x)

#期望、均值 表示为E(x)
mean1 = np.sum(x)/x.size
mean2 = np.mean(x)
print('期望%s'%mean1)
print('期望%s'%mean2)

#方差 表示变量与数据期望之间的偏离程度
var1 = np.sum((x-mean1)**2)/x.size
var2=np.cov(x)
var_y=np.cov(y)
print('方差%s'%var1)
print('方差%s'%var2)
print('方差%s'%var_y)

#标准差
sd1 = np.sqrt(var1)
sd2 = np.std(x)
print('标准差%s'%sd1)
print('标准差%s'%sd2)

#协方差 0.21501753就是协方差的值,因为是N-1的原因
cov1=np.cov(x,y,ddof=0) #1：无偏估计，0：简单平均
cov2=np.mean(np.multiply(x,y))-(np.mean(x)*np.mean(y))
print('协方差矩阵%s'%cov1)
print('协方差%s'%cov2)

#协方差图形展示
x1 =np.array([8.0,5.9,4.0,6.1,7.5,9.5,7.2])
x2 =np.array([5.0,3.3,0.8,2.9,4.0,6.0,3.9])
x3 =np.array([7.0,8.0,8.5,5.6,3.0,2.5,3.1])
cov_x1x2=np.cov(x1,x2,ddof=0)
cov_x1x3=np.cov(x1,x3,ddof=0)
#相关系数
cov1_x1x2 = np.corrcoef(x1,x3)
cov2_x1x2 = np.cov(x1,x3,ddof=0)/(np.std(x1)*np.std(x3))
print('协方差矩阵cov1_x1x2%s'%cov1_x1x2)
print('协方差矩阵cov2_x1x2%s'%cov2_x1x2)
print('协方差矩阵x1x2%s'%cov_x1x2)
print('协方差矩阵x1x3%s'%cov_x1x3)
#画下图
import matplotlib.pylab as plb
plb.figure(figsize=(13, 6), facecolor='w')
plb.subplot(2,1,1)
plb.plot(x1,'ro-',linewidth='2',label='x1',zorder=2)
plb.plot(x2,'bo-',linewidth='2',label='x2',zorder=3)
plb.plot(x3,'ko-',linewidth='2',label='x3',zorder=4)
plb.legend(loc='lower right')
plb.ylabel(u'协方差相关性', fontsize=12)

#大数定理
def generate_random_int(n):
    """产生n个1~10的随机数"""
    return [np.random.randint(1,10) for i in range(n)]
number = 5000
#设定X坐标值
x_theorem=[i for i in range(number+1) if i !=0]
#创建数组
total_random_int = generate_random_int(number)
#设定Y坐标值 平均数
y_theorem=[np.mean(total_random_int[0:i+1]) for i in range(number)]

plb.subplot(2,1,2)
plb.plot(x_theorem,y_theorem,'b-',label='期望值')
plb.xlim(0,number/10)
plb.grid(True)
plb.ylabel(u'大数定理', fontsize=12)
plb.legend(loc='lower right')
plb.show()

'''
loc：float
    此概率分布的均值（对应着整个分布的中心centre）
scale：float
    此概率分布的标准差（对应于分布的宽度，scale越大越矮胖，scale越小，越瘦高），就是方差开根号，seq和(x-u)**2/n
size：int or tuple of ints
    输出的shape，默认为None，只输出一个值
'''
n=10000
b_values = np.random.normal(loc=-1.0, scale=2.0, size=n)
plb.figure(facecolor='w')
plb.hist(b_values, bins=100, color='#FF0000')
plb.show()

#最大似然估计
