from ctypes import *
import os
import numpy as np

LIB_PATH = os.path.abspath("..\\..\\lib\\linear-model\\cmake-build-debug\\ML-framework.dll")
LIB_PATH2 = os.path.abspath("..\\..\\lib\\multilayer-perceptron\\cmake-build-release-visual-studio\\ML-framework.dll")


def load_simple_perceptron_ml_library():
    library = CDLL(LIB_PATH)

    library.createModel.argtypes = [c_int]
    library.createModel.restype = c_void_p

    library.saveModel.argtypes = [c_void_p, c_int, c_char_p]
    library.saveModel.restype = c_bool

    library.loadModel.argtypes = [c_char_p]
    library.loadModel.restype = c_void_p

    library.predictClassificationModel.argtypes = [c_void_p, c_void_p, c_int]
    library.predictClassificationModel.restype = c_double

    library.predictRegressionModel.argtypes = [c_void_p, c_void_p, c_int]
    library.predictRegressionModel.restype = c_double

    library.trainModel.restype = None
    library.trainModel.argtypes = [c_void_p, c_void_p, c_void_p, c_int, c_int, c_int,
                                   c_double]

    library.deleteModel.argtypes = [c_void_p]
    library.deleteModel.restype = None
    return library


def load_multilayer_perceptron_ml_library():
    library = CDLL(LIB_PATH2)

    library.createModel.argtypes = [c_int]
    library.createModel.restype = c_void_p

    library.addLayer.argtypes = [c_void_p, c_int, c_char_p]
    library.addLayer.restype = None

    library.summary.argtypes = [c_void_p]
    library.summary.restype = None

    library.predict.argtypes = [c_void_p, c_void_p]
    library.predict.restype = POINTER(c_double * 3)

    library.trainModel.restype = None
    library.trainModel.argtypes = [c_void_p, c_void_p, c_void_p, c_int, c_int, c_double]

    library.deleteModel.argtypes = [c_void_p]
    library.deleteModel.restype = None
    return library


def s(string):
    return c_char_p(string.encode())


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]


def to_native(X, Y):
    XFlattened = np.reshape(X, len(X) * len(X[0]))
    labels = (c_double * len(Y))(*list(Y))
    inputs = (c_double * len(XFlattened))(*list(XFlattened))
    return inputs, labels


def linear_model_accuracy(library, network, X, Y):
    correct = 0
    for i, value in enumerate(X):
        input_test = (c_double * len(value))(*list(value))
        prediction = library.predictRegressionModel(network, input_test, len(input_test))
        correct += 1 if np.sign(prediction) == Y[i] else 0
    return correct / len(X) * 100


def mlp_regression_accuracy(library, network, X, Y):
    correct = 0
    for i, value in enumerate(X):
        input_test = (c_double * len(value))(*list(value))
        prediction = library.predict(network, input_test).contents[0]
        correct += 1 if find_nearest(Y, prediction) == Y[i] else 0
    return correct / len(X) * 100


def mlp_classification_accuracy(library, network, X, Y, label_count):
    correct = 0
    for i, value in enumerate(X):
        input_test = (c_double * len(value))(*list(value))
        predictions = [library.predict(network, input_test).contents[p] for p in range(0, label_count)]
        correct += 1 if np.argmax(predictions) == Y[i] else 0
    return correct / len(X) * 100
