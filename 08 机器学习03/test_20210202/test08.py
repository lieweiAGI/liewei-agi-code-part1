#加载红酒数据
import numpy as np

wine_data = np.loadtxt("数据集/红酒数据/wine.data",delimiter=",")
x = wine_data[:,1:]
y = wine_data[:,0]
print(x)
print(y)