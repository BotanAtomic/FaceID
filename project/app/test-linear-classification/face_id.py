# Created by botan on 5/22/2020.

from ctypes import *


def predict_face(ml_lib, networks, data):
    inputData = (c_double * len(data))(*list(data))
    predictions = {}
    for key, network in networks.items():
        predictions[key] = ml_lib.predictRegressionModel(network, inputData, len(data))

    return predictions, max(predictions, key=predictions.get)
