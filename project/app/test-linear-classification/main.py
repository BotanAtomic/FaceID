# Created by botan on 5/4/2020.

import matplotlib.pyplot as plt

from camera_stream import start_camera_stream
from dataset import *
from face_id import *
from library_api import *

img_size = 48
img_channel = 3
img_total_size = (img_size * img_size * img_channel)

LIB_PATH = os.path.abspath("..\\..\\lib\\cmake-build-debug\\ML-framework.dll")
train_percent = 0.85

liveTest = True

networks = {}

os.makedirs("..\\..\\models\\face-classification", exist_ok=True)

(X, Y, X_test, Y_test, labels) = load_dataset(img_total_size, train_percent)
XFlattened = np.reshape(X, len(X) * len(X[0]))
allExamplesInputs = (c_double * len(XFlattened))(*list(XFlattened))

ml_lib = load_ml_library(LIB_PATH, img_total_size, len(XFlattened), len(Y))

print("Successfully loaded ML library")

for label in labels:
    save_path_model = os.path.abspath("..\\..\\models\\face-classification\\" + label + ".model")
    labelId = labels.index(label)
    model = None

    if os.path.exists(save_path_model):
        print('Restore neural network for label', label, ' id = ', labelId)
        model = ml_lib.loadModel(c_char_p(save_path_model.encode()))

    if model is None:
        print('Create and training neural network for label', label, ' id = ', labelId)
        normalizedY = Y.copy()

        for i in range(0, len(normalizedY)):
            if normalizedY[i] == labelId:
                normalizedY[i] = 1
            else:
                normalizedY[i] = -1

        model = ml_lib.createModel(len(X[0]))
        allExamplesExpectedOutputs = (c_double * len(Y))(*list(normalizedY))
        ml_lib.trainModel(model, allExamplesInputs, allExamplesExpectedOutputs, len(Y), img_total_size, 10000, 0.01)
        ml_lib.saveModel(model, len(X[0]), c_char_p(save_path_model.encode()))

    networks[label] = model

correct = 0
for i in range(0, len(X_test)):
    data = X_test[i]
    must_predict = labels[Y_test[i]]
    (predictions, best_match) = predict_face(ml_lib, networks, data)
    if best_match == must_predict:
        correct += 1
    else:
        plt.imshow(np.reshape(data, (48, 48, 3)))
        plt.show()
        print('Mismatch', predictions, must_predict)

print('Accuracy = ', (correct / len(Y_test) * 100), ' -- ', correct, '/', len(Y_test))

if liveTest:
    start_camera_stream(ml_lib, networks, img_total_size)

for network in networks.values():
    ml_lib.deleteModel(network)
