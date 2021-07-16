#多任务岭回归

from sklearn import linear_model
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score,explained_variance_score


x, y = datasets.make_regression(n_samples=100, n_features=1, n_targets=3,
noise=10, random_state=0)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
#多任务回归模型
reg = linear_model.MultiTaskLasso()
reg.fit(x_train, y_train)

y_pred = reg.predict(x_test)
# 平均绝对误差
print(mean_absolute_error(y_test, y_pred))
# 均方误差
print(mean_squared_error(y_test, y_pred))
# R2 评分
print(r2_score(y_test, y_pred))
# explained_variance
print(explained_variance_score(y_test, y_pred))
_x = np.array([-2.5, 2.5])
_y = reg.predict(_x[:,None])
plt.scatter(x_test, y_test[:,0],color="green")
plt.plot(_x, _y[:,0], linewidth=3, color="orange")
plt.scatter(x_test, y_test[:,1],color="blue")
plt.plot(_x, _y[:,1], linewidth=3, color="red")
plt.scatter(x_test, y_test[:,2],color="yellow")
plt.plot(_x, _y[:,2], linewidth=3, color="black")
plt.show()