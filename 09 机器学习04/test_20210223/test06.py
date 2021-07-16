#降维
#案例：将鸢尾花数据特征分布可视化

import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA

#加载数据
data = load_iris()
x = data.data
y = data.target
print(x)
print(y)

#加载PCA算法，设置降维后主成分的数目（目标维度）
pca = PCA(n_components=2)
#对数据进行降维操作
reduced_x = pca.fit_transform(x)
print(reduced_x)

#将降维后的特征可视化
color = ["r","g","b"]
for i,label in enumerate(y):
    plt.scatter(reduced_x[i][0],reduced_x[i][1],c=color[label])
plt.show()