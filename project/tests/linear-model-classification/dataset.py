# Created by botan on 5/22/2020.

import os
import glob
import numpy as np
from PIL import Image


def load_dataset(img_total_size, train_percent):
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