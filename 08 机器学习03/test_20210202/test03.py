#普通线性回归
from sklearn import linear_model,svm
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error,mean_squared_error,explained_variance_score,r2_score
import numpy as np

#加载数据
x,y = datasets.make_regression(n_samples=100,n_features=1,noise=10)
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3)

#加载模型
#线性回归模型
# reg = linear_model.LinearRegression()
#岭回归
# reg = linear_model.Ridge(0.2)
#Lasso回归
# reg = linear_model.Lasso(0.2)
#弹性网络
# reg = linear_model.ElasticNet(0.3,0.3)
#贝叶斯回归
reg = linear_model.BayesianRidge()
# #SVR
# reg = svm.SVR(C=13, gamma=0.1)
#模型拟合
reg.fit(x_train,y_train)
#评估
pred_y = reg.predict(x_test)
#平均绝对误差
print(mean_absolute_error(y_test,pred_y))
#均方误差
print(mean_squared_error(y_test,pred_y))
#可解释方差
print(explained_variance_score(y_test,pred_y))
#R2得分
print(r2_score(y_test,pred_y))

#可视化
_x = np.array([-2.5,2.5])
_y = reg.predict(_x[:,None])

plt.scatter(x_test,y_test)
plt.plot(_x,_y,linewidth=3,color="red")
plt.show()