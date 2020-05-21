from ctypes import *
import numpy as np
import os
import glob
from PIL import Image

img_size = 48
img_channel = 3
img_size = (img_size * img_size * img_channel)

LIB_PATH = os.path.abspath("..\\..\\lib\\cmake-build-debug\\ML-framework.dll")
train_percent = 0.8


def load_ml_library(totalSize, exampleSize):
    library = CDLL(LIB_PATH)

    library.createModel.argtypes = [c_int]
    library.createModel.restype = c_void_p

    library.predictClassificationModel.argtypes = [c_void_p, c_double * img_size, c_int]
    library.predictClassificationModel.restype = c_double

    library.trainModel.restype = None
    library.trainModel.argtypes = [c_void_p, c_double * totalSize, c_double * exampleSize, c_int, c_int, c_int, c_double]

    library.deleteModel.argtypes = [c_void_p]
    library.deleteModel.restype = None
    return library


def load_dataset():
    labels = [os.path.basename(x) for x in filter(
        os.path.isdir, glob.glob(os.path.join("..\\..\\dataset\\train", '*')))]

    X = []
    Y = []

    X_test = []
    Y_test = []

    for label in labels:
        files = [os.path.abspath(x) for x in filter(
            os.path.isfile, glob.glob(os.path.join("..\\..\\dataset\\train\\" + label, '*')))]
        total = len(files)
        index = 0
        for imageFile in files:
            img_array = np.array(Image.open(imageFile)) / 255.0
            img_array = np.reshape(img_array, 48 * 48 * 3)
            if index / total < train_percent:
                X.append(img_array)
                Y.append(labels.index(label))
            else:
                X_test.append(img_array)
                Y_test.append(labels.index(label))
            index += 1

    print('Loading dataset with labels:', labels)
    return X, Y, X_test, Y_test, labels


networks = {}

(X, Y, X_test, Y_test, labels) = load_dataset()
XFlattened = np.reshape(X, len(X) * len(X[0]))
allExamplesInputs = (c_double * len(XFlattened))(*list(XFlattened))

ml_lib = load_ml_library(len(XFlattened), len(Y))

print("Successfully loaded ML library")

for label in labels:
    labelId = labels.index(label)
    print('Create neural network for label', label, ' id = ', labelId)
    normalizedY = Y.copy()

    for i in range(0, len(normalizedY)):
        if normalizedY[i] == labelId:
            normalizedY[i] = 1
        else:
            normalizedY[i] = -1

    allExamplesExpectedOutputs = (c_double * len(Y))(*list(normalizedY))

    model = ml_lib.createModel(len(X[0]))
    networks[label] = model

    print("Training model")

    ml_lib.trainModel(model, allExamplesInputs, allExamplesExpectedOutputs, len(Y), img_size, 40000, 0.01)

correct = 0
for i in range(0, len(X_test)):
    data = X_test[i]
    person = Y_test[i]
    inputData = (c_double * len(data))(*list(data))
    predicted = ''
    must_predict = labels[Y_test[i]]
    for key, network in networks.items():
        response = int(ml_lib.predictClassificationModel(network, inputData, len(data)))
        if response == 1:
            predicted = key

    if predicted == must_predict:
        correct += 1
    else:
        print('Mismatch between', predicted, 'and', must_predict)

print(correct, '/', len(Y_test))
print('Accuracy = ', (correct / len(Y_test) * 100))
