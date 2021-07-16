import numpy as np
from sklearn.kernel_ridge import KernelRidge
import matplotlib.pyplot as plt

#给随机数添加种子，让每次随机的结果都一致。
rng = np.random.RandomState(0)
x = 5 * rng.rand(100, 1)
y = np.sin(x).ravel()
print(y)
# 添加噪声
y[::5] += 3 * (0.5 - rng.rand(x.shape[0] // 5))
#创建模型
kr = KernelRidge(kernel="rbf",gamma=0.1)
#模型拟合
kr.fit(x,y)
#创建测试数据
X_plot = np.linspace(0, 5, 100)
y_kr = kr.predict(X_plot[:, None])
plt.scatter(x, y)#数据可视化
plt.plot(X_plot, y_kr, color="red")#模型可视化
plt.show()
# print("========================")
# print(y[::5])