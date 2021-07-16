#数据预处理
from sklearn import preprocessing
import numpy as np

# x = np.array([[1.,-300,2],
#               [2,0,0],
#               [0,1,-3],
#               [2,1,300]])
# #标准化
# # 将每一列特征标准化为标准正态分布。注意：标准化针对的是数据的特征，而不是批次。
# x_scale = preprocessing.scale(x)
# print(x_scale)
# # print(x_scale.mean(0),x_scale.std(0))
# #
# # scaler = preprocessing.StandardScaler().fit(x)
# # print(scaler)
# # scaler_x = scaler.transform(x)
# # print(scaler_x)
#
# #minmax
# scaler = preprocessing.MinMaxScaler().fit(x)
# print(x)
# # print(scaler)
# # scaler_x = scaler.transform(x)
# # print(scaler_x)
# # print(scaler_x.mean(0),scaler_x.std(0))
#
# #MaxAbsScaler
# scaler = preprocessing.MaxAbsScaler().fit(x)
# print(scaler)
# scaler_x = scaler.transform(x)
# print(scaler_x)
#
# #RobustScaler注意：数据中包含大量异常值时，用这个函数来做预处理
# scaler = preprocessing.RobustScaler().fit(x)
# scaler_x = scaler.transform(x)
# print(scaler_x)
# #正则化Normalization
# scaler = preprocessing.Normalizer("l2").fit(x)
# scaler_x = scaler.transform(x)
# print(scaler_x)
# #二值化
# scaler = preprocessing.Binarizer().fit(x)
# scaler_x = scaler.transform(x)
# print(scaler_x)



