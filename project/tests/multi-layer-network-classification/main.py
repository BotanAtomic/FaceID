# Created by botan on 5/4/2020.
from camera_stream import start_camera_stream
from dataset import *
from library_api import *

train_percent = 0.95

live_test = False
force_train = False
test_dataset = True

if train_percent >= 1:
    test_dataset = False

(X, Y, X_test, Y_test, labels) = load_dataset(train_percent)
XFlattened = np.reshape(X, len(X) * len(X[0]))

ml_lib = load_ml_library(len(labels))

network = ml_lib.createModel(len(X[0]))

ml_lib.addLayer(network, 256, cstring("activation=sigmoid"))  # hidden layer
ml_lib.addLayer(network, 128, cstring("activation=sigmoid"))  # hidden layer
ml_lib.addLayer(network, 128, cstring("activation=sigmoid"))  # hidden layer
ml_lib.addLayer(network, len(labels), cstring("activation=sigmoid"))  # output layer

inputs = (c_double * len(XFlattened))(*list(XFlattened))
ml_lib.trainModel(network, inputs, (c_double * len(Y))(*list(Y)), len(Y), 150, 0.01)

ml_lib.saveModel(network, cstring("neuralNetwork.model"))

if test_dataset:
    accuracy = analyse_dataset(ml_lib, network, X_test, Y_test)
    print('Neural network accuracy:', accuracy)

if live_test:
    start_camera_stream(ml_lib, network)

ml_lib.deleteModel(network)


