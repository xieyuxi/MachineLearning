# -*- coding: utf-8 -*-
"""
Created on Thu Jan 26 00:32:26 2017

@author: admin
"""

import KNN
# matplotlib的pyplot子库提供了和matlab类似的绘图API，方便用户快速绘制2D图表。
import matplotlib
import matplotlib.pyplot as plt
from numpy import *

#==============================================================================
# group, labels = KNN.createDataSet()
#
# result = KNN.classify0([0,0], group, labels, 3)
#
# print result
#==============================================================================

filename = "datingTestSet.txt"

datingDataMat, datingLabels = KNN.file2matrix(filename)
# 这里创建了一个新的 figure，我们要在上面作图
fig = plt.figure()
# add_subplot(111)相当于在figure上添加子图
# 其中 111 的意思是，将子图切割成 1行 1列的图，并在第 1 块上作画
# 所以这里只有一副画
# 再举个例子 add_subplot(22x)，将子图切割成 2行 2列的图，并在第 X 块上作画
ax = fig.add_subplot(111)
# 利用scatter()函数分别取已经处理好的矩阵datingDataMat的第一列和第二列数据
# 并用不同颜色将类别标识出来
ax.scatter(datingDataMat[:,0], datingDataMat[:,1],
15.0*array(datingLabels), 15.0*array(datingLabels))
# 展示我们做好的 figure
plt.show()

KNN.autoNorm(datingDataMat)