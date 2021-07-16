#加载房价数据
import numpy as np

house_data = np.loadtxt("数据集/房价数据/housing.data")
print(house_data.shape)
x = house_data[:,:13]
y = house_data[:,-1]
