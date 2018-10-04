# -*- coding:utf-8 -*-
import numpy as np
import matplotlib.pylab as plb
from sympy import *
import matplotlib as mpl

mpl.rcParams['font.sans-serif']=[u'simHei']
mpl.rcParams['axes.unicode_minus']=False

x = Symbol("x")
#f =sin(x)
f =np.e**x
df = []
for n in range(21):
    df.append(diff(f,x,n).subs(x,0))
def taylorFunction(x,n):
    """
    泰勒展开式
    :param x:
    :param n:
    :return:
    """
    if n==0:
        return df[0]
    return taylorFunction(x,n-1)+df[n]/np.math.factorial(n)*x**n

x1=np.arange(-4*np.pi/2,4*np.pi/2,0.5)
#y1=np.sin(x1)
y1=np.e**x1
plb.plot(x1,y1,'r',linewidth='2',label='原函数',zorder=2)
plb.plot(x1,taylorFunction(x1,5),'go-',linewidth='2',label='相似函数',zorder=1)

plb.legend(loc='lower right')
plb.show()