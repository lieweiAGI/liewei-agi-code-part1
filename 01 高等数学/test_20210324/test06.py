import random
import matplotlib.pyplot as plt
import time

_x = [i/100 for i in range(100)]
print(_x)
_y = [3*e+4+random.random() for e in _x]
print(_y)

#定义参数w和b
w = random.random()
b = random.random()

for i in range(30):
    for x, y in zip(_x, _y):
        z = w * x + b  # 前向计算
        # 根据输出，得到损失
        o = z - y
        loss = o ** 2

        dw = -2 * o * x
        db = -2 * o

        w = w + dw * 0.5
        b = b + db * 0.5
        print(w, b, loss)



# plt.plot(_x,_y,".")
# v = [w*e+b for e in _x]
# plt.plot(_x,v)
# plt.show()