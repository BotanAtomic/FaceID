import matplotlib.pyplot as plt
import numpy as np
from ctypes import *

from utils import *

X = np.array([[1, 0], [0, 1], [0, 0], [1, 1]])
Y = np.array([1, 1, -1, -1])

plt.scatter(X[0:2, 0], X[0:2, 1], color='blue')
plt.scatter(X[2:4, 0], X[2:4, 1], color='red')
plt.show()
plt.clf()

# ON PASSE AUX CHOSES SERIEUSES

output = 1

ml_lib = load_multilayer_perceptron_ml_library(output)

network = ml_lib.createModel(len(X[0]))
ml_lib.addLayer(network, 5, cstring("activation=tanh"))  # hidden layer
ml_lib.addLayer(network, output, cstring("activation=tanh"))  # output layer

XFlattened = np.reshape(X, len(X) * len(X[0]))
labels = (c_double * len(Y))(*list(Y))
inputs = (c_double * len(XFlattened))(*list(XFlattened))

ml_lib.trainModel(network, inputs, labels, len(labels), 1000, 0.01)


# Test
def test(input_test, label):
    prediction = ml_lib.predict(network, (c_double * len(input_test))(*list(input_test))).contents[0]
    print(prediction)
    if prediction > 0 and label == 1:
        return 1
    elif prediction <= 0 and label == -1:
        return 1
    else:
        print("INCORRECT PREDICTION", prediction, label)
    return 0


correctPrediction = 0
for i, value in enumerate(X):
    correctPrediction += test(value, Y[i])

print("Accuracy:", correctPrediction / len(X) * 100, "%")
