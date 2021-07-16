#KNN算法简单调参
from sklearn import neighbors,datasets,preprocessing
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

#引入数据
iris = datasets.load_iris()
x = iris.data
y = iris.target
#数据预处理
scaler = preprocessing.StandardScaler().fit(x)
x = scaler.transform(x)
print(x)
#设置超参范围1~30
k_range = range(1,31)
k_score = []
for k in k_range:
    #创建模型
    knn = neighbors.KNeighborsClassifier(n_neighbors=k,algorithm="kd_tree")
    scores = cross_val_score(knn,x,y,cv=5,scoring="accuracy")
    print(scores.mean())
    k_score.append(scores.mean())
    print("=============================")
#可视化结果
plt.figure()
plt.plot(k_range,k_score)
plt.show()