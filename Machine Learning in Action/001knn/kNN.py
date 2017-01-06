from numpy import *
import operator


def createDataSet():
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


def classify(inX, dataSet, labels, k):
    # shape函数是numpy.core.fromnumeric中的函数，它的功能是读取矩阵的长度
    dataSetSize = dataSet.shape[0]

    # 对输入inX进行二维扩展，先扩展为[x,y],再扩展为dataSetSize个[x,y],从后往前扩展
    # 计算两个矩阵的差
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet

    # **表示乘方,对矩阵的差进行乘方计算
    sqDiffMat = diffMat ** 2

    # 将一个矩阵的每一行向量相加 距离公式：distance^2 = (x1-x2)^2 + (y1-y2)^2
    sqDistances = sqDiffMat.sum(axis=1)
    # distance = (x1-x2)^2 + (y1-y2)^2 开根号
    distances = sqDistances ** 0.5

    # argsort函数返回的是数组值从小到大的索引值
    sortedDistIndicies = distances.argsort()

    # 一个类似于Map<String,Integer>的Map
    # 用于统计每个Label对应出现的次数
    classCount = {}
    # range(5)  # 代表从0到5(不包含5)
    for i in range(k):
        voteILabel = labels[sortedDistIndicies[i]]
        classCount[voteILabel] = classCount.get(voteILabel, 0) + 1

    # 对Map按照key值排序
    # iterable指定要排序的list或者iterable
    # 指定取待排序元素的哪一项进行排序
    # reverse是一个bool变量，表示升序还是降序排列，默认为false（升序排列），定义为True时将按降序排列。
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

# 测试
group, labels = createDataSet()
result = classify([0, 0], group, labels, 3)
print(result) #B
