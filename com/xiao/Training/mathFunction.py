# -*- coding:utf-8 -*-
import matplotlib.pylab as plb
import numpy as np
import matplotlib as mpl
import math

#设置图形中文乱码
mpl.rcParams['font.sans-serif']=[u'simHei']
mpl.rcParams['axes.unicode_minus']=False

#取值范围0.05~3，步长为0.05，个数为(3%0.05）-1
x=np.arange(0.05,3,0.05)
x1 = np.linspace(-np.pi, np.pi, 100)

plb.figure(figsize=(13, 6), facecolor='w')
#常函数
y1=5+0*x
plb.plot(x,y1,linewidth='2',label='常函数:y=5')

#1次函数
y2=1+2*x
plb.plot(x,y2,linewidth='2',label='1次函数:y=1+2x')

#2次函数
y3=1.5*x**2-2+1
plb.plot(x,y3,linewidth='2',label='2次函数:y=1.5$x^2$-2x-1')

#幂函数
y4=x**2
plb.plot(x,y4,linewidth='2',label='幂函数:y=$x^2$')

#指数函数
y5=np.e**x
plb.plot(x,y5,linewidth='2',label='指数函数:y=$e^x$')

#对数函数
y6=[math.log(i,0.5) for i in x]
plb.plot(x,y6,linewidth='2',label='对数函数:y=log0.5(x)')

plb.legend(loc='lower right')
plb.grid(True)
plb.show()

'''
from sympy import *
x = Symbol("x")
y = Symbol("y")
m=diff(log(x,10)*y+sqrt(y+x),y)
print(m)
'''

