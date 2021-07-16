import matplotlib.pyplot as plt
import numpy

x = numpy.random.randint(0,100,20)
y = numpy.random.randint(0,100,20)
x1 = numpy.random.randint(0,100,20)
y1 = numpy.random.randint(0,100,20)

plt.scatter(x,y,c="r",marker=".",s=20, label="boy")
plt.scatter(x1,y1,c="g",marker="v", s=20,label="girl")
plt.legend() #显示图例
plt.show()