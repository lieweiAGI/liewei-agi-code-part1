#自动调参
import numpy as np
from sklearn.kernel_ridge import KernelRidge
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV#自动调参矩阵

#给随机数添加种子，让每次随机的结果都一致。
rng = np.random.RandomState(0)
x = 5 * rng.rand(100, 1)
y = np.sin(x).ravel()
print(y)
# 添加噪声
y[::5] += 3 * (0.5 - rng.rand(x.shape[0] // 5))
#创建模型
# kr = KernelRidge(kernel="rbf",gamma=0.1)
kr = GridSearchCV(KernelRidge(),
                  param_grid={"kernel": ["rbf", "laplacian", "polynomial","sigmoid"],
                  "alpha": [1e0, 0.1, 1e-2, 1e-3],
                  "gamma": np.logspace(-2, 2, 20)})
#模型拟合
kr.fit(x,y)
#创建测试数据
X_plot = np.linspace(0, 5, 100)
#自动调参的结果（效果最好的参数）
print(kr.best_score_, kr.best_params_)
# y_kr = kr.predict(X_plot[:, None])
# plt.scatter(x, y)#数据可视化
# plt.plot(X_plot, y_kr, color="red")#模型可视化
# plt.show()
# print("========================")
# print(y[::5])