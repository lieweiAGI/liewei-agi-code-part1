#构建回归数据
import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_regression
#x为样本特征，y为样本输出,coef为回归系数，一共生成100条样本，每个样本有1个特征
x,y,coef = make_regression(n_samples=100,n_features = 1,noise=10,coef=True)
print(x)
print(y)

#画图
plt.scatter(x,y)
plt.plot(x,x*coef,color = "blue",linewidth=3)
plt.show()