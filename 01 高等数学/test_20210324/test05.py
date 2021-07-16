import numpy as np
import matplotlib.pyplot as plt

x = np.arange(-10,10,0.1)
sigmoid = 1/(1+np.exp(-x))
tanh = (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))

plt.plot(x,sigmoid)
plt.plot(x,tanh)
plt.show()