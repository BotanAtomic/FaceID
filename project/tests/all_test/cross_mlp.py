import matplotlib.pyplot as plt
import numpy as np
from utils import *

X = np.random.random((500, 2)) * 2.0 - 1.0
Y = np.array([1 if abs(p[0]) <= 0.3 or abs(p[1]) <= 0.3 else -1 for p in X])

plt.scatter(np.array(list(map(lambda elt: elt[1], filter(lambda c: Y[c[0]] == 1, enumerate(X)))))[:, 0],
            np.array(list(map(lambda elt: elt[1], filter(lambda c: Y[c[0]] == 1, enumerate(X)))))[:, 1], color='blue')
plt.scatter(np.array(list(map(lambda elt: elt[1], filter(lambda c: Y[c[0]] == -1, enumerate(X)))))[:, 0],
            np.array(list(map(lambda elt: elt[1], filter(lambda c: Y[c[0]] == -1, enumerate(X)))))[:, 1], color='red')
plt.show()
plt.clf()

output = 1

ml_lib = load_multilayer_perceptron_ml_library(output)

network = ml_lib.createModel(len(X[0]))
ml_lib.addLayer(network, 16, cstring("activation=relu"))    # hidden layer 1
ml_lib.addLayer(network, 8, cstring("activation=sigmoid"))  # hidden layer 2
ml_lib.addLayer(network, 4, cstring("activation=sigmoid"))  # hidden layer 3
ml_lib.addLayer(network, 2, cstring("activation=sigmoid"))  # hidden layer 4
ml_lib.addLayer(network, output, cstring("activation=tanh"))  # output layer

XFlattened = np.reshape(X, len(X) * len(X[0]))
labels = (c_double * len(Y))(*list(Y))
inputs = (c_double * len(XFlattened))(*list(XFlattened))

ml_lib.trainModel(network, inputs, labels, len(labels), 3000, 0.01)

ml_lib.summary(network)

# Test
def test(input_test, label):
    prediction = ml_lib.predict(network, (c_double * len(input_test))(*list(input_test))).contents[0]

    if prediction > 0 and label == 1:
        return 1
    if prediction <= 0 and label == -1:
        return 1

    return 0


correctPrediction = 0
for i, value in enumerate(X):
    correctPrediction += test(value, Y[i])

print("Accuracy:", correctPrediction / len(X) * 100, "%")
