import kNN
from numpy import array
import matplotlib as mpl
import matplotlib.pyplot as plt

datingDataMat, datingLabels = kNN.file2matrix('datingTestSet2.txt')
print(datingDataMat)
print(datingLabels[0:20])

# 设置中文字体
mpl.rcParams['font.sans-serif'] = ['SimHei']
fig = plt.figure()
# 将画布分割成1行1列，图像画在从左到右从上到下的第1块
ax = fig.add_subplot(111)
plt.xlabel("玩视频游戏所耗时间百分比")
plt.ylabel("每周消耗的冰淇淋公升数")
# scatter表示绘制散点图
# plot表示绘制折线图
# ax.scatter(datingDataMat[:, 1], datingDataMat[:, 2])
# 第3和第4个参数分别表示点的大小和颜色
# 默认的颜色映射表中蓝色与最小值对应，红色与最大值对应。
# ax.scatter(datingDataMat[:, 1], datingDataMat[:, 2], 15.0 * array(datingLabels), 15.0 * array(datingLabels))
# plt.show()

# 数据归一化处理
noreMat, ranges, minVals = kNN.autoNorm(datingDataMat)
print(noreMat)
print(ranges)
print(minVals)
ax.scatter(noreMat[:, 1], noreMat[:, 2], 15.0 * array(datingLabels), 15.0 * array(datingLabels))
plt.show()
