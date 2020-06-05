import matplotlib.pyplot as plt
import numpy as np
from ctypes import *

from utils import load_simple_perceptron_ml_library



X = np.concatenate(
    [np.random.random((50, 2)) * 0.9 + np.array([1, 1]), np.random.random((50, 2)) * 0.9 + np.array([2, 2])])
Y = np.concatenate([np.ones((50, 2)), np.ones((50, 2)) * -1.0])

print(Y)
print(X)
plt.scatter(X[0:50, 0], X[0:50, 1], color='blue')
plt.scatter(X[50:100, 0], X[50:100, 1], color='red')
plt.show()
plt.clf()

print(Y)