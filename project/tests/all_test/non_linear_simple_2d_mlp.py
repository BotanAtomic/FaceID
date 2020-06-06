import matplotlib.pyplot as plt
import numpy as np
from ctypes import *

from utils import *

X = np.array([
    [1],
    [2],
    [3]
])
Y = np.array([
    2,
    3,
    2.5
])

plt.scatter(X, Y)
plt.show()
plt.clf()

output = 1

ml_lib = load_multilayer_perceptron_ml_library(output)

network = ml_lib.createModel(len(X[0]))
ml_lib.addLayer(network, 12, cstring("activation=tanh;initializer=random_uniform,0,1"))  # output layer
ml_lib.addLayer(network, 8, cstring("activation=tanh;initializer=random_uniform,0,1"))  # output layer
ml_lib.addLayer(network, output, cstring("activation=relu;initializer=random_uniform,0,1"))  # output layer

XFlattened = np.reshape(X, len(X) * len(X[0]))
labels = (c_double * len(Y))(*list(Y))
inputs = (c_double * len(XFlattened))(*list(XFlattened))

ml_lib.trainModel(network, inputs, labels, len(labels), 8000, 0.1)


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]


# Test
def test(input_test, label):
    prediction = ml_lib.predict(network, (c_double * len(input_test))(*list(input_test))).contents[0]
    print(prediction, label)
    return 1 if find_nearest(Y, prediction) == label else 0


correctPrediction = 0
for i, value in enumerate(X):
    correctPrediction += test(value, Y[i])

print("Accuracy:", correctPrediction / len(X) * 100, "%")
