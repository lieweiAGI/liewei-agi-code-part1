#K_Means聚类
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

#加载数据
data = np.loadtxt("数据集/test.txt")
# plt.scatter(data[:,0],data[:,1])
# plt.show()
#创建模型
clf = KMeans(n_clusters=4,max_iter=1000)
#模型拟合
clf.fit(data)
#预测
clf.predict(data)
#聚类的中心
print(clf.cluster_centers_)
#每个数据所属的类别
labels = clf.labels_
print(labels)
#用来评估每一堆数据与自己中心点的距离和。距离越小，就说明聚类的效果越好。
print(clf.inertia_)

#可视化
clolor = ["red","blue","green","black"]
for index,label in enumerate(labels):
    plt.scatter(data[:,0][index],data[:,1][index],c=clolor[label])
plt.show()