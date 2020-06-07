# Created by botan on 5/22/2020.

import os
import glob
import numpy as np
from PIL import Image
from const import *
import matplotlib.pyplot as plt

from face_id import predict_face


def load_dataset(train_percent):
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
            img_array = np.reshape(img_array, img_total_size)
            if index / total < train_percent:
                X.append(img_array)
                Y.append(labels.index(label))
            else:
                X_test.append(img_array)
                Y_test.append(labels.index(label))
            index += 1

    print('Loading dataset with labels:', labels)
    return X, Y, X_test, Y_test, labels


def analyse_dataset(ml_lib, network, X_test, Y_test):
    correct = 0
    for i in range(0, len(X_test)):
        data = X_test[i]
        must_predict = Y_test[i]
        (predictions, best_match) = predict_face(ml_lib, network, data)
        if best_match == must_predict:
            correct += 1
        else:
            plt.imshow(np.reshape(data, (img_size, img_size, img_channel)))
            plt.show()
            print('Mismatch', predictions, must_predict)
    return correct / len(Y_test) * 100
