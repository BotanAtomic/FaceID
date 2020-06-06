from ctypes import *
import os

LIB_PATH = os.path.abspath("..\\..\\lib\\linear-model\\cmake-build-debug\\ML-framework.dll")
LIB_PATH2 = os.path.abspath("..\\..\\lib\\multilayer-perceptron\\cmake-build-debug-visual-studio\\ML-framework.dll")


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


def load_multilayer_perceptron_ml_library(outputSize):
    library = CDLL(LIB_PATH2)

    library.createModel.argtypes = [c_int]
    library.createModel.restype = c_void_p

    library.addLayer.argtypes = [c_void_p, c_int, c_char_p]
    library.addLayer.restype = None

    library.summary.argtypes = [c_void_p]
    library.summary.restype = None

    library.predict.argtypes = [c_void_p, c_void_p]
    library.predict.restype = POINTER(c_double * outputSize)

    library.trainModel.restype = None
    library.trainModel.argtypes = [c_void_p, c_void_p, c_void_p, c_int, c_int, c_double]

    library.deleteModel.argtypes = [c_void_p]
    library.deleteModel.restype = None
    return library


def cstring(string):
    return c_char_p(string.encode())
