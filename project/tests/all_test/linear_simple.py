import matplotlib.pyplot as plt
import numpy as np
from ctypes import *

from utils import load_simple_perceptron_ml_library

X = np.array([
    [1, 1],
    [2, 3],
    [3, 3]
])
Y = np.array([
    1,
    -1,
    -1
])

plt.scatter(X[0, 0], X[0, 1], color='blue')
plt.scatter(X[1:3, 0], X[1:3, 1], color='red')
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
    input_test = (c_double * len(input_test))(*list(input_test))
    prediction = ml_lib.predictRegressionModel(model, input_test, len(input_test))
    return 1 if np.sign(prediction) == label else 0


correctPrediction = 0
for i, value in enumerate(X):
    correctPrediction += test(value, Y[i])

print("Accuracy:", correctPrediction / len(X) * 100, "%")

ml_lib.deleteModel(model)  # Parce qu'on est des gens bien
