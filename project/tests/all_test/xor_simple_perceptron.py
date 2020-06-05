import matplotlib.pyplot as plt
import numpy as np
from ctypes import *

from utils import load_simple_perceptron_ml_library

X = np.array([[1, 0], [0, 1], [0, 0], [1, 1]])
Y = np.array([1, 1, -1, -1])

plt.scatter(X[0:2, 0], X[0:2, 1], color='blue')
plt.scatter(X[2:4, 0], X[2:4, 1], color='red')
plt.show()
plt.clf()

ml_lib = load_simple_perceptron_ml_library()

model = ml_lib.createModel(len(X[0]))

XFlattened = np.reshape(X, len(X) * len(X[0]))
labels = (c_double * len(Y))(*list(Y))
inputs = (c_double * len(XFlattened))(*list(XFlattened))

ml_lib.trainModel(model, inputs, labels, len(Y), len(X[0]), 1000, 0.01)

# Test
def test(input_test, label):
    prediction = ml_lib.predictRegressionModel(model, (c_double * len(input_test))(*list(input_test)), len(input_test))
    print(input_test, label, prediction)
    return 1 if np.sign(prediction) == label else 0


correctPrediction = 0
for i, value in enumerate(X):
    correctPrediction += test(value, Y[i])

print("Simple perceptron accuracy:", correctPrediction / len(X) * 100, "%")