# Created by botan on 5/4/2020.


from camera_stream import start_camera_stream
from dataset import *
from library_api import *

LIB_PATH = os.path.abspath("..\\..\\lib\\linear-model\\cmake-build-debug\\ML-framework.dll")
train_percent = 0.80

live_test = False
force_train = False
test_dataset = True

(X, Y, X_test, Y_test, labels) = load_dataset(train_percent)
XFlattened = np.reshape(X, len(X) * len(X[0]))
allExamplesInputs = (c_double * len(XFlattened))(*list(XFlattened))

ml_lib = load_ml_library(len(labels))

network = ml_lib.createModel(len(X[0]))

ml_lib.addLayer(network, 512, cstring("activation=sigmoid;"))  # hidden layer
ml_lib.addLayer(network, 512, cstring("activation=sigmoid;"))  # hidden layer
ml_lib.addLayer(network, 3, cstring("activation=sigmoid;"))  # output layer

inputs = (c_double * len(XFlattened))(*list(XFlattened))
ml_lib.trainModel(network, inputs, (c_double * len(Y))(*list(Y)), len(Y), 60, 0.1)

if test_dataset:
    accuracy = analyse_dataset(ml_lib, network, X_test, Y_test)
    print('Neural network accuracy:', accuracy)

#Dernier test 100% (possible que le model ne generalise pas)

if live_test:
    start_camera_stream(ml_lib)

ml_lib.deleteModel(network)
