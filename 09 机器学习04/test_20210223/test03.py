#One_hot编码
import numpy as np

#设置类别数量
num_class = 10
#需要编码的数据（标签）
target = np.array([3,6,7,9,2,1])
#做法一：
one_hot = np.zeros((len(target),num_class))
print(one_hot)
# target = target.reshape(-1,1)
print(target)
for i in range(one_hot.shape[0]):
    index = target[i]
    one_hot[i][index] = 1
print(one_hot)
#做法二
#将整数转为一个num_class的one-hot编码
one_hot = np.eye(10)[target]
print(one_hot)

#可以通过argmax()将one-hot编码还原
print(np.argmax(one_hot,axis=1))