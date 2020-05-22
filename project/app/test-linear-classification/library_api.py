# Created by botan on 5/22/2020.

from ctypes import *


def load_ml_library(LIB_PATH, img_total_size, totalSize, exampleSize):
    library = CDLL(LIB_PATH)

    library.createModel.argtypes = [c_int]
    library.createModel.restype = c_void_p

    library.saveModel.argtypes = [c_void_p, c_int, c_char_p]
    library.saveModel.restype = c_bool

    library.loadModel.argtypes = [c_char_p]
    library.loadModel.restype = c_void_p

    library.predictClassificationModel.argtypes = [c_void_p, c_double * img_total_size, c_int]
    library.predictClassificationModel.restype = c_double

    library.predictRegressionModel.argtypes = [c_void_p, c_double * img_total_size, c_int]
    library.predictRegressionModel.restype = c_double

    library.trainModel.restype = None
    library.trainModel.argtypes = [c_void_p, c_double * totalSize, c_double * exampleSize, c_int, c_int, c_int,
                                   c_double]

    library.deleteModel.argtypes = [c_void_p]
    library.deleteModel.restype = None
    return library
