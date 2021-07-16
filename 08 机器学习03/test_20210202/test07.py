#加载房价数据
#基础架构
from sklearn import neighbors, datasets, preprocessing,svm
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import r2_score
import joblib
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GridSearchCV#自动调参矩阵

house_data = np.loadtxt("数据集/房价数据/housing.data")
print(house_data.shape)
x = house_data[:,:13]
y = house_data[:,-1]
#划分训练集与测试集
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3)
#数据预处理
scaler = preprocessing.StandardScaler().fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

#创建模型
kr = svm.SVR()
# kr = KernelRidge(kernel="rbf",gamma=0.1)
# kr = GridSearchCV(KernelRidge(),
#                   param_grid={"kernel": ["rbf", "laplacian", "polynomial","sigmoid","linear"],
#                   "alpha": [1e0, 0.1, 1e-2, 1e-3],
#                   "gamma": np.logspace(-2, 2, 20)})
#
#模型拟合
kr.fit(x_train,y_train)
# #自动调参的结果（效果最好的参数）
# print(kr.best_score_, kr.best_params_)
# #保存模型
# joblib.dump(kr,"params/kr.pkl")
#模型加载
# kr = joblib.load("params/kr.pkl")
y_pred = kr.predict(x_test)
print(y_pred)
print(r2_score(y_test,y_pred))
