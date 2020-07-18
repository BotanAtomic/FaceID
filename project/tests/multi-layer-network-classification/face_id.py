# Created by botan on 5/22/2020.

from ctypes import *
import numpy as np


def predict_face(ml_lib, network, data):
    inputData = (c_double * len(data))(*list(data))
    predictions = [p for p in ml_lib.predict(network, inputData).contents]
    print(predictions)
    return predictions, np.argmax(predictions)
