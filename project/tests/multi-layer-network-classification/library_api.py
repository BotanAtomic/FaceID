# Created by botan on 5/22/2020.

from ctypes import *
import os


LIB_PATH2 = os.path.abspath("..\\..\\lib\\multilayer-perceptron\\cmake-build-release-visual-studio\\ML-framework.dll")

def load_ml_library(outputSize):
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

