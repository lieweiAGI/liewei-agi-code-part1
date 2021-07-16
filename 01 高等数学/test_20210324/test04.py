import numpy as np
import matplotlib.pyplot as plt

x = np.arange(0.1,10,0.1)

y1 = np.log(x)
y2 = np.log(x)/np.log(np.array([0.5]))
plt.plot(x,y1)
plt.plot(x,y2)
plt.show()