# -*- coding: utf-8 -*-
"""
Created on Tue Feb 07 13:11:50 2017

@author: xie yuxi
"""

from math import log
import operator

def createDataSet():
    # dataSet里面每一列都是一个属性特征，每一行表示一个实例，里面的值为属性值
    dataSet = [[1, 1, 'yes'],
               [1, 2, 'yes'],
               [1, 3, 'no'],
               [0, 4, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing','flippers']
    #change to discrete values
    return dataSet, labels

""" 计算传入的 dataSet 的熵
1、先求出这个dataSet中所有类别出现的频率：
        如： 类别 A 出现的频率 P（A） = A出现的次数 / 所有类别出现的总次数
2、根据熵的计算公式获得这个dataSet的信息熵：
        H = - P(A)*log2P(A) - P(B)*log2P(B) - P(C)*log2P(C) -....-P(X)*log2P(X)

Args:
    dataSet ： 需要计算熵的数据集

Attention：
    实例：就是 List 中的一条数据

Returns:
     这个数据集的熵

Raises:
    None
"""
def calcShannonEnt(dataSet):
    # numEntries 表示 dataSet 的实例总数（也表示所有类别出现的次数）
    numEntries = len(dataSet)
    # 声明一个字典 labelCounts 用于存放 {label(类别) ： label(类别)出现的次数}
    labelCounts = {}
    # 遍历 dataSet 这个数据集中的每一个实例
    for featVec in dataSet:
        # currentLabel 取的是每一个实例的最后一列(也就是每个实例的类别)作为节点属性
        currentLabel = featVec[-1]
        # 如果取出的类别不在 labelCounts 中时，就将该属性值出现的次数初始化为 0
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        # 如果取出的属性值存在于 labelCounts 字典中时，就将该类别出现次数加 1
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    # 遍历每个 labelCounts 字典里的 key值，目的是为了获得每个实例中各类别的出现频率
    # 这里的Key指的是类别名
    for key in labelCounts:
        # 计算每个类别出现频率 = 该类别出现次数 / 这个数据集中所有类别出现的总次数
        # 计算的时候需要将labelCounts[key]做float类型的强制转换
        prob = float(labelCounts[key])/numEntries
        # 根据熵的公式，计算这个数据集的熵：
        #       H = - P(A)*log2P(A) - P(B)*log2P(B) -....- P(X)*log2P(X)
        # 其中 log(prob,2)表示以 2为底 prob 的对数
        shannonEnt -= prob * log(prob,2) #log base 2
    # 返回这个数据集的熵
    return shannonEnt



"""将传入的数据集 dataSet 按 axis 属性列的具体值 value 拆出一个子数据集 subdataset
1、先建立一个新的 List 用于存储新的子数据集
2、将 axis 列等于 value 的值的实例抽取出来插入新子数据集 retDataSet

Args:
    dataSet ： 待划分的父数据集
    axis ： 一个实例的某一列
    value ：一个实例 axis 列对应的值

Returns:
       axis 列等于 value 的实例集合（同时实例集的 axis 列被剔除）

Raises:
    None
"""
def splitDataSet(dataSet, axis, value):
    # 构建一个新的 List 用于存放特征列 axis 值为 value 的实例集
    retDataSet = []
    # 遍历每一个在 dataSet 中的实例 featVec
    for featVec in dataSet:
        # 如果特征列 axis 的值为 value，则取出该实例
        if featVec[axis] == value:
            # 首先取这个符合条件的实例的前半段(以 axis 为界，剔除 axis 特征这一列)
            reducedFeatVec = featVec[:axis]
            # 再取出符合条件的实例的后(以 axis 为界，剔除 axis 特征这一列)
            # 得到一个既符合判断条件，又剔除了 axis 这一列的新实例 reducedFeatVec
            reducedFeatVec.extend(featVec[axis+1:])
            # 将实例 reducedFeatVec 添加进新建的 List retDataSet中
            retDataSet.append(reducedFeatVec)
    # 返回所有符合
    return retDataSet


""" 计算信息增益并取最优的特征属性作为非子叶节点来将数据集分类
1、先求出传入的整个数据集 dataSet 的熵 baseEntropy
    - 公式：H(baseEntropy) = - P(A)*log2P(A) - P(B)*log2P(B) -....-P(X)*log2P(X)
    - 可通过函数 calcShannonEnt()计算数据集的熵
2、再计算出每个特征属性(实例中除最后一列类别外的所有特征属性列)的熵 newEntropy
    - 每个特征属性的熵 等于 这个属性下面每个值的期望条件熵的和
    - 举个栗子：假设计算特征属性 A 的熵：
        - A 属性下可能会有不同的值，按每个值将数据集分为多个子数据集
            - 比如 A 属性下有: X1、X2 和 X3三个值
            - 首先将数据集按特征属性 A 取 X1时划分子数据集，并计算这个子数据集的熵
            - 同样的方式计算特征属性 A 取 X2 和 X3 所划分数据集的熵
            - 则特征属性 A 的熵则是这几个分数据集的熵的期望值
            - 即 H(A) = P(X1)H(A|X1) + P(X2)H(A|X2) + P(X3)H(A|3)
                - 以计算 P(X1)H(A|X1) 为例：
                - 比如假设 A 取 X1 划分出了子集 SubDataSet1
                - SubDataSet1 子集中共有 n1 个实例
                - 权数 P(X1) = n1 / S (S为 传入数据集dataSet的总共实例数)
                - SubDataSet1的熵 H(A|X1) = calcShannonEnt(SubDataSet1)
                -
（  * 期望值和平均数的区别：
    - 平均数是一个统计学概念，均数是实验后根据实际结果统计得到的样本的平均值
    - 期望是一个概率论概念，期望是实验前根据概率分布“预测”的样本的平均值
    - 当统计数量很小的数群时，我们用所有实验结果的和除以实验对象数量就可以得到平均数
    - 但当这个数群的数量很大时,我们只好做个抽样,并“期望”透过抽样所得到的均值,
      去预测整个群体的“期望值”.
    - 期望可以理解为加权平均值之和，而权数就是函数的密度(即某个值的取值概率).
 ）
3、计算信息增益(infoGain)，挑选最好的特质做节点
    - 信息增益表示排除不确定性的能力，增益越大，说明越能将杂乱变顺
    - 信息增益 = 原数据集的熵 H(dataSet) - 子数据集的熵（比如：H(A)）
4、返回使增益最大的特征属性的在实例中的位置序号

Args:
    dataSet ： 需要计算熵的数据集

Attention：
    实例：就是 List 中的一条数据

Returns:
     返回最优特殊属性所在实例 List 的序号值

Raises:
    None
"""
def chooseBestFeatureToSplit(dataSet):
    # 取特征属性的个数（这里实例的最后一列是类别，所以减 1 ）
    numFeatures = len(dataSet[0]) - 1
    # 首先计算总体数据集的熵
    baseEntropy = calcShannonEnt(dataSet)
    # 声明最大增益值，并初始化为 0.0
    bestInfoGain = 0.0;
    # 声明最优特征值的序号，并初始化为 -1
    bestFeature = -1
    # range(numFeatures) 生成长度为 numFeatures 的数列，i依次赋值 0 ~ numFeatures
    # 遍历所有的特征属性
    for i in range(numFeatures):
        # featList 用来存取获得的第i个特征所有的可能的取值（这里取的是列值）
        # [example[i] for example in dataSet] 这种写法等同于：
        #       for example in dataSet:
        #           featList.append(example[i])
        #       这里的 example 是一个实例，包含了所有特征属性的列
        #       example[i]表示的是这一个实例第 i 列特征属性的取值，并存入featList
        featList = [example[i] for example in dataSet]
        # set(List)函数返回列表中不同的字符串所组成的集合uniqueVals，方便比对计算
        uniqueVals = set(featList)
        # 初始化各特征值的熵
        newEntropy = 0.0
        # 遍历特征属性序号为 i 所有取值，用这个值做 value 将数据集划分子数据集
        # 划分出的子集所包含的所有实例数目之和等于父集的实例数目
        for value in uniqueVals:
            # subDataSet 只是 i 等于某一个具体值 value 所划分出来的子集
            subDataSet = splitDataSet(dataSet, i, value)
            # prob为这个子集所包含的实例数在总数据集实例数中占比
            prob = len(subDataSet)/float(len(dataSet))
            # newEntropy 就是 i 特征属性的熵
            # 即所有 i 可能取值划分的子集的期望值
            # 举个栗子：i = A 且 A 取 X1， X2，X3
            # 则 newEntropy = H(A) = P(X1)H(A|X1) + P(X2)H(A|X2) + P(X3)H(A|3)
            newEntropy += prob * calcShannonEnt(subDataSet)
        # 计算信息增益，就是总数据集的熵减去某个特征属性的熵之差，就是减少的不确定性
        # 增益越大，代表减少的不确定性就越大，也就是分类能力越强，增益最大化来选特征
        infoGain = baseEntropy - newEntropy
        # 每次将不同特征对应的信息增益和原来的最大增益做比较，信息增益永远取最大一个
        if (infoGain > bestInfoGain):
            # 如果现在的增益比上一个最大的增益还大，就把这个赋值为当前最大的一个增益
            bestInfoGain = infoGain
            # 然后将增益最大的这个特征的序号赋值给变量 bestFeature
            bestFeature = i
    # 仅仅返回最大信息增益对应的特征属性的序号值
    return bestFeature

""" 选出传入的类别列表中出现次数最多的类别
1、先建立一个新字用于存储 {类别 ： 出现次数}
2、遍历类别列表，如果类别不在字典 classCount 中时，类别的出现次数初始化为 0，
                    如果类别在字典中，则该类别的出现次数加 1
3、将字典中出现的  {类别 ： 出现次数} 按出现的次数正序排序
4、选出其中出现次数最多的一个类别

Args:
    classList ： 传入的类别列表（注意这里传入的是类别列表）

Returns:
       出现次数最多对应的类别

Raises:
    None
"""
def majorityCnt(classList):
    # 建立新字典用于存放 {特征标签 ： 出现次数} 对
    classCount={}
    # 遍历特征列表中的特征
    for vote in classList:
        # 如果特征不在字典classCount中时
        if vote not in classCount.keys():
            # 将其添加进字典，并且出现次数初始化为0
            classCount[vote] = 0
        # 如果存在，则该特征出现次数加1
        classCount[vote] += 1
    # 将获得的classCount字典变为可迭代类型，并按照第一个域，也就是出现次数排序
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    # 返回排名第一的 {特征标签 ： 出现次数} 对子中的特征标签
    return sortedClassCount[0][0]


""" 构建决策树
1、根据传入的数据集，先找出最优特征标签作父节点
2、接着特征的具体取值看做分支条件,将数据划分为不同的子数据集（分支）
3、采用递归自己调用自己，为分支再找最优特征，再在分支上建立分支
4、直到要不所有剩下的实例都是一类，或者特征只剩下一个，停止递归，并返回构件好的决策树

Args:
    dataSet ： 传入需要构建决策树的数据集
    labels ： 传入的数据集对应的特征标签

Returns:
       返回构件好的决策树

Raises:
    None
"""
def createTree(dataSet,labels):
    # 新建一个列表来存储类别
    # 以下等同于：
    # for example in dataSet:
    #       classList.append(example[-1])
    classList = [example[-1] for example in dataSet]
    # 如果当classList中的所有类别都一样时，则停止划分子数据集
    # count()函数用于统计字符串里某个字符出现的次数并返回该字符串出现次数
    if classList.count(classList[0]) == len(classList):
        # 当 classList 中都是一个类别时，就返回这个类别，做决策树的叶子
        # 因为此时 classList 中的类别都一样，所以选第一个做代表 classList[0]
        return classList[0]
    # 如果当传入数据集只剩一个特征，即只剩一列时，但是这一列里还有不同的类别
    # 也就是所有标签都用完还没有把剩下的实例分好类，那么就返回出现次数最多的类别
    if len(dataSet[0]) == 1:
        # 调用函数 majorityCnt()，并返回出现次数最多的类别做这个判断分支的类别
        return majorityCnt(classList)
    # 通过函数 chooseBestFeatureToSplit() 从传入的 dataSet中选出最优特征的序号
    bestFeat = chooseBestFeatureToSplit(dataSet)
    # 通过序号在传入的 Labels 中找到对应的特征标签
    bestFeatLabel = labels[bestFeat]
    # 构建决策树，字典中包含字典，每一层都选最优标签做父节点
    myTree = {bestFeatLabel:{}}
    # 标签使用过一次后就将其从 Labels 中删除，因为一个特征标签只能用一次
    del(labels[bestFeat])
    # 这里是需要找到得出来的最好的特征标签下所有的取值，为划分子数据集准备
    # 以下等同于：
    #       for example in dataSet:
    #           featValues.append(example[bestfeat])
    featValues = [example[bestFeat] for example in dataSet]
    # 构建一个 set 来储存最优特征下的所取得值（如果有多个值，只留一个）
    uniqueVals = set(featValues)
    # 遍历这个最优特征下取到的不重复的值
    for value in uniqueVals:
        # 建立新的 subLabels 来拷贝原 labels, 这样不会影响原来的标签
        # 此时的 subLabels 剔除了本轮最优标签
        subLabels = labels[:]
        # 这里是使用递归调用自己，不停地为每个分支再创建分支，直到两个if中的一个出现
        #       - 要不是剩余实例都是一个类别
        #       - 要不是标签只剩一个，但是类还没分出来
        # 其中先使用splitDataSet（）函数将传入数据集按标签和取值分类，传入子标签
        # createTree()函数返回的是一个判断路径走到底的类别（叶子节点）
        # myTree[bestFeatLabel][value]结合决策树图形来理解：
        #       - bsetFeatLabel 是树的节点，value是决策树判断的分支
        #           - 如果某个分支已经分好了类别，满足if条件，
        #           则 myTree[bestFeatLabel][value] 就返回这个分支的类别（叶子）
        #           - 如果分支还没有分好类，就递归再调用自己再找最有标签分类
        #           - 直到符合if条件之一，这样每个分支最终都是一个类别（叶子）
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value),subLabels)
    return myTree


myDat,labels = createDataSet()

myTree = createTree(myDat,labels)