import numpy

a = numpy.arange(48)
print(a)
a = a.reshape(4,4,3)
print(a)
b = a.reshape(2,2,4,3) #切高
print(b[0])
b = b[0].reshape(-1)
print(b)
print("====================")
c = a.reshape(2,4,2,3) #切宽
print(c[0])
c = c[0].reshape(-1)
print(c)