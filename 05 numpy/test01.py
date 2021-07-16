import numpy as np

x = np.array([1,2,3], dtype=np.float32)
y = [1.,2.,3.]
print(type(x))
print(x.dtype)
print(type(y[0]))

#列表生成数组
a = [[[1,2,3],[1,2,3],[1,2,3]]]
x = np.array(a)
print(x)
print(x.shape) #形状
print(x.ndim) #维度
print(x.size) #元素个数

#特殊矩阵 ones/zeros/empty 1/0/空
x = np.ones([3,3,3], dtype=np.int8)
print(x.dtype)
x = np.zeros([2])
print(x.dtype)
x = np.empty([2,2])
print(x.dtype)

#使用arange生成连续的元素
x = np.arange(10)
x = np.arange(2,10)
print(x)
x = np.arange(2,10,1)
print(x)
print(type(x))
print(x.dtype)

#astype复制数组,创建副本
x = np.arange(10, dtype=np.float32)
print(x.dtype)
y = x.astype(dtype=np.int8)
print(x)
print(y)
print(y.dtype)

#字符串转数字
x = np.array(['1','2','3','4','5'])
print(x)
print(x.dtype)
y = x.astype(dtype=np.float64)
print(y)
print(y.dtype)

#数组的运算
# x = np.array([[1,2,3,4,5],[1,2,3,4,5]])
# y = np.array([[5,4,3,2,1],[1,2,3,4,5]])
# print(x+y) #加法
# print(x*y) #数乘
# print(x.dot(y),"===============")
# x= np.array([[1,2],[3,4]])
# y = np.array([[5,6,],[7,8]])
# print(x)
# print(y)
# print(x.dot(y)) #点乘
# print(y.dot(x)) #点乘

x = np.array([1,2,3,4,5])
print(x>2) #生成布尔类型的数组，掩码

#基本索引
x = np.array([9,8,7,6,5,4,3,2,1,0])
print(x)
print(x[9])
x = np.array([[1,2,3,],[4,5,6]])
print(x)
print(x[1][2])
print(x[1,2]) #通常用
print(x.ndim)
print(x)
print(x[0].ndim)
print(x[0])
print(x[0][0].ndim) #索引会降维
print(x[0][0])

x = np.array([[[1,2],[3,4]],[[5,6],[7,8]]])
print(x.shape)
print(x)
print(x[1,0,0])
y = x[1,0].copy()  #生成一个副本
print(y)
z = x[1,0]
y[1] = 0
z[1] = 0
print(y)
print(z)
print(x)

#切片
x = np.array([1,2,3,4,5])
print(x[:])
print(x[2:])
print(x[:2])
print(x[1:4])
print(x[1:4:2])
print(x[1:x.size])
print(x[0:x.size])

x = np.array([[1,2,3,4,5],[6,7,8,9,0]])
print(x.ndim)
print(x[:1])
print(x[:1].ndim) #切片不会降维
print(x.shape)
print(x[:,1:4])
print(x[:,4:])

#布尔索引
x = np.arange(5)
y = np.array([False,False,False,True,True,])
print(x)
print(y)
print(x[y])
print(x[x>2])
print(x[y==False])
print(x[~(x>2)])

#数组索引
x = np.array([1,2,3,4,5,6])
print(x)
y = np.array([-1,-2,-3])
print(x[y])
print(x[x[0:3]])
x = np.array([[1,2],[3,4],[5,6],[7,8]])
print(x[[0,1]])
print(x[0,1])
print(x[[0,1],[0,1]])

#轴
#reshape
x = np.array([[[1,2],[3,4]],[[5,6],[7,8]],[[9,0],[1,2]]])
print(x.shape)
print(x)
print(x.reshape((2,2,3)))
print(x.reshape((2,2,3)).shape)

x = np.arange(8).reshape((2,4))
print(x)
print(x.T)

#轴变换
x = np.arange(12).reshape((2,2,3))
print(x)
k = x.transpose(2,1,0) #轴变换
print(k)
k = x.swapaxes(0,2) #轴交换
print(k)

#基本统计
x = np.arange(10)
print(x)
print(x.mean())
print(x.max())
print(x.min())
print(x.sum())

x = np.arange(12).reshape((2,6))
print(x)
print(x.max(axis=1))
print(x.sum(axis=0))
print(x.mean(axis=0))

#where
x = np.arange(6)
x = np.array([3,3,3,3,3])
y = np.where(x>2,9,1) #掩码
print(x)
print(y)

#存取
x = np.arange(100).reshape((4,5,5))
np.save("file",x) #以二进制的方式存储
y = np.load("file.npy")
print(y)

#矩阵求逆
x = np.array([[1,1],[1,2]])
y = np.linalg.inv(x)
print(x)
print(y)
print(x.dot(y)) #单位矩阵

#随机数
# x = np.random.randn() #服从正态分布的
# x = np.random.randint() #随机整数
# x = np.random.random() #0~1的随机值
# x = np.random.seed() #随机数的种子
x = np.random.randint(0,2,10000) #模拟抛硬币
print((x>0).sum()) #正面次数

#广播
x = np.arange(12).reshape((2,2,3))
y = np.array([1,2,3])
print(x)
print(x)
print(y)
print(x*y)