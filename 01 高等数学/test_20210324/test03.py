import numpy as np
import matplotlib.pyplot as plt

x = np.arange(-3,3,0.1)
print(x)

y = np.power(0.5,x)
print(y)

y1 = np.power(2,x)

plt.plot(x,y)
plt.plot(x,y1)
plt.show()