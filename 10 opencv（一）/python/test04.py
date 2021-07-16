import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import numpy

ax = numpy.random.normal(0, 1, 50)
ay = numpy.random.normal(0, 1, 50)
az = numpy.random.normal(0, 1, 50)

fig = plt.figure()
a = Axes3D(fig)
a.scatter(ax,ay,az,c="green",marker="x", s=50)
plt.legend()
plt.show()