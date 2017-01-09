from numpy import *
import operator


def createDataSet():
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


# inX表示预测的点
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


# 将文件中的数据转换为二维矩阵
def file2matrix(filename):
    fr = open(filename)
    arrayOLines = fr.readlines()
    numberOfLines = len(arrayOLines)
    # 创建全为0的二维矩阵
    returnMat = zeros((numberOfLines, 3))
    classLabelVector = []
    index = 0
    for line in arrayOLines:
        # s为字符串，rm为要删除的字符序列
        # 当rm为空时，默认删除空白符（包括'\n', '\r',  '\t',  ' ')
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index, :] = listFromLine[0:3]
        # 下标-1表示最后一个元素
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    return returnMat, classLabelVector


# 数据归一化
# 取消某一项特征或属性数据量纲过大导致的对整体数据的影响过大
# newValue = (oldValue-min)/(max - min)
# norm:规范
def autoNorm(dataSet):
    # 获取每一列的最大值和最小值
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals, (m, 1))
    normDataSet = normDataSet / tile(ranges, (m, 1))
    return normDataSet, ranges, minVals


# 约会测试类方法
#  错误率计算
def datingClassErrorRateTest():
    hoRatio = 0.10
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m * hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify(normMat[i, :], normMat[numTestVecs:m, :], datingLabels[numTestVecs:m], 5)
        print("the classifier came back with:%d, the real answer is:%d" % (classifierResult, datingLabels[i]))
        if (classifierResult != datingLabels[i]):
            errorCount += 1.0

    print("the total error rate is %f" % (errorCount / float(numTestVecs)))


# 约会网站预测函数
def classifyPerson():
    resultList = ['not at all', 'in small doses', 'in large doses']
    percentTats = float(input("percentage of time spent playing video games?"))
    ffMiles = float(input("frequent flier miles earned per year?"))
    iceCream = float(input("liters of ice cream consumed per year?"))

    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)

    inArr = array([ffMiles, percentTats, iceCream])
    classifierResult = classify((inArr - minVals) / ranges, normMat, datingLabels, 3)
    print("You will probably like this person:", resultList[classifierResult - 1])
