import numpy as np
import matplotlib.pyplot as plt

x = np.arange(0.1,10,0.1)
print(x)

y = np.tile(np.array([3]),x.shape)
print(y)

plt.plot(x,y)
plt.show()