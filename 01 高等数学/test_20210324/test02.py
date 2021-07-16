import numpy as np
import matplotlib.pyplot as plt

x = np.arange(-3,3,0.1)
print(x)

y = np.power(x,2)
print(y)

x1 = np.arange(0,3,0.1)
y1 = np.power(x1,0.5)

y2 = np.power(x,3)
# plt.plot(x,y)
# plt.plot(x1,y1)
plt.plot(x,y2)
plt.show()
