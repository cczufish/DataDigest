import trees
import treePlotter

myDat, labels = trees.createDataSet()
print(myDat)

shannonEnt = trees.calcShannonEnt(myDat)
# 0.9709505944546686
print(shannonEnt)

trees.splitDataSet(myDat, 0, 1)

bestFeature = trees.chooseBestFeatureToSplit(myDat)
print('bestFeature=%d' % bestFeature)
# bestFeature=0

myTree = trees.createTree(myDat, labels)
print(myTree)

treePlotter.createPlot()
