# -*- coding:utf-8 -*-
'''
梯度下降代码示例

'''
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import math
from mpl_toolkits.mplot3d import Axes3D
from sympy import *

# 解决中文显示问题
mpl.rcParams['font.sans-serif'] = [u'SimHei']
mpl.rcParams['axes.unicode_minus'] = False

'''
通过梯度下降法求函数极值点
'''
X = []
Y = []
Z = []

x1 = -3
y1 = 1

x = Symbol("x")
y = Symbol("y")
f = x ** 2 + y ** 2  #原函数
f_change = f.subs({x:x1,y:y1})
f_current = f.subs({x:x1,y:y1})
step = 0.1
X.append(x1)
Y.append(y1)
Z.append(f_current)
while f_change > 1e-10: #1e-10 = 10^-10
    #函数沿着它的导函数方向增长/或减少的速度最快，
    # 因为这个是凸函数所以沿其负导数方向的函数减小最快，有最小值
    x1 = x1 - step * diff(f,x,1).subs(x,x1)
    y1 = y1 - step * diff(f,y,1).subs(y,y1)
    f_change = f_current - f.subs({x:x1,y:y1})
    f_current = f.subs({x:x1,y:y1})
    X.append(x1)
    Y.append(y1)
    Z.append(f_current)
print (u"最终结果为:", (x1, y1))

#画图
X2 = np.arange(-2, 2, 0.1)
Y2 = np.arange(-2, 2, 0.1)
X2, Y2 = np.meshgrid(X2, Y2)
Z2 = X2 ** 2 + Y2 ** 2

fig = plt.figure()
ax = Axes3D(fig)
#构建一个x,y-2到2的曲面
ax.plot_surface(X2, Y2, Z2, rstride=1, cstride=1, cmap='rainbow')
#标注点
ax.plot(X, Y, Z, 'ro:')

ax.set_title(u'梯度下降法求解, 最终解为: x=%.2f, y=%.2f, z=%.2f' % (x1, y1, f_current))

plt.show()

"""
线条风格linestyle或ls	描述	线条风格linestyle或ls	描述
‘-‘	实线	
‘:’	虚线	 
‘–’	破折线	 
‘-.’	点划线	 
线条标记
‘o’	圆圈	‘.’	点
‘D’	菱形	‘s’	正方形
‘h’	六边形1	‘*’	星号
‘H’	六边形2	‘d’	小菱形
‘_’	水平线	‘v’	一角朝下的三角形
‘8’	八边形	‘<’	一角朝左的三角形
‘p’	五边形	‘>’	一角朝右的三角形
‘,’	像素	‘^’	一角朝上的三角形
‘+’	加号	‘\	‘	竖线
‘None’,’’,’ ‘	无	‘x’	X
颜色
别名	颜色	别名	颜色
b	蓝色	g	绿色
r	红色	y	黄色
c	青色	k	黑色	 
m	洋红色	w	白色
"""
