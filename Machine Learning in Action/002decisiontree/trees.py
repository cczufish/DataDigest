# ------------------------决策树算法------------------------------
from math import log
import operator


# 计算给定数据集的香农熵
def calcShannonEnt(dataSet):
    # 计算数据集中实例的总数
    numEntries = len(dataSet)
    # 创建一个数据字典，用于存放每一个目标的对应的特征属性出现的次数
    labelCounts = {}
    for featVec in dataSet:
        # 键值是最后一列的数值（最后一列是目标属性，是否购买，是否是真实账户）
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key]) / numEntries
        shannonEnt -= prob * log(prob, 2)
    return shannonEnt


def createDataSet():
    # 定义一个列表
    dataSet = [
        [1, 1, 'yes'],
        [1, 1, 'yes'],
        [1, 0, 'no'],
        [0, 1, 'no'],
        [0, 1, 'no'],
    ]
    labels = ['no surfacing', 'flippers']
    return dataSet, labels


# dataSet 待划分的数据集
# axis 划分数据集的特征
# value 特征的返回值(问题的目标变量的值)
def splitDataSet(dataSet, axis, value):
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            # 假设根据第3个属性进行划分，则先获取第1,2个属性
            reducedFeatVec = featVec[:axis]
            # 再获取第4,5，……后面的属性，排除掉第三个属性
            # 将整个集合划分为一个维度更小的集合 降维
            reducedFeatVec.extend(featVec[axis + 1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet


# 根据当前数据集进行一次分割
# 数据的最后一列代表当前实例的类别标签
def chooseBestFeatureToSplit(dataSet):
    # - 1 的目的是减去目标属性
    numFeatures = len(dataSet[0]) - 1

    # 计算原始的整个数据集的香农熵
    baseEntropy = calcShannonEnt(dataSet)

    bastInfoGain = 0.0
    bestFeature = - 1
    for i in range(numFeatures):
        featList = [example[i] for example in dataSet]
        uniqueVals = set(featList)
        newEntropy = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet) / float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)
        infoGain = baseEntropy - newEntropy
        if (infoGain > bastInfoGain):
            bastInfoGain = infoGain
            bestFeature = i
    return bestFeature


def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys(): classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

#递归创建树
def createTree(dataSet, labels):
    classList = [example[-1] for example in dataSet]
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel: {}}
    del (labels[bestFeat])
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)
    return myTree
