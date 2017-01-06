#使用knn算法对电影的分类进行预测

import numpy as np
from sklearn import neighbors

knn = neighbors.classification.KNeighborsClassifier()  # 取得knn分类器
data = np.array([[3, 104], [2, 100], [1, 81], [101, 10], [99, 5], [98, 2]])  # data对应着打斗次数和接吻次数
labels = np.array([1, 1, 1, 2, 2, 2])  # labels则是对应Romance和Action
knn.fit(data, labels)  # 导入数据进行训练
#result = knn.predict([18, 90])
result = knn.predict([90, 90])
print(result)
