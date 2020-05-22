from ctypes import *
import numpy as np

print("Creation du Dataset de Test")

X = np.array([
    [1, 1],
    [2, 3],
    [3, 3]
])
Y = np.array([
    1,
    -1,
    -1
])


print("Chargement de la DLL et definition des signatures des méthodes")
my_dll_path = "F:/2020-3A-IABD-DemoPACppLib/DemoPACppLib/x64/Debug/DemoPACppLib.dll"

ml_lib = CDLL(my_dll_path)

ml_lib.create_linear_model.argtypes = [c_int]
ml_lib.create_linear_model.restype = c_void_p
ml_lib.predict_linear_model_regression.argtypes = [c_void_p, c_double * 2, c_int]
ml_lib.predict_linear_model_regression.restype = c_double
ml_lib.predict_linear_model_classification.argtypes = [c_void_p, c_double * 2, c_int]
ml_lib.predict_linear_model_classification.restype = c_double
ml_lib.train_linear_model_regression.argtypes = [c_void_p, c_double * 6, c_double * 3, c_int, c_int]
ml_lib.train_linear_model_regression.restype = None
ml_lib.train_linear_model_classification.argtypes = [c_void_p, c_double * 6, c_double * 3, c_int, c_int, c_double,
                                                     c_int]
ml_lib.train_linear_model_classification.restype = None
ml_lib.delete_linear_model.argtypes = [c_void_p]
ml_lib.delete_linear_model.restype = None


print("Creation du modèle et test de prédiction sur les valeurs du premier exemple")
inputs = (c_double * 2)(*list(X[0]))
model = ml_lib.create_linear_model(2)
predictedOutput = ml_lib.predict_linear_model_regression(model, inputs, 2)

print(predictedOutput)
predictedOutput = ml_lib.predict_linear_model_classification(model, inputs, 2)

print(predictedOutput)


print("Entrainement de notre modèle")
XFlattened = np.reshape(X, 3 * 2)
allExamplesInputs = (c_double * 6)(*list(XFlattened))
allExamplesExpectedOutputs = (c_double * 3)(*list(Y))
ml_lib.train_linear_model_classification(model, allExamplesInputs, allExamplesExpectedOutputs,
                                         3, 2, 0.1, 42)

print("Test de Prediction après entrainement")
predictedOutput = ml_lib.predict_linear_model_regression(model, inputs, 2)

print(predictedOutput)
predictedOutput = ml_lib.predict_linear_model_classification(model, inputs, 2)

print(predictedOutput)

print("Suppression de notre modele")
ml_lib.delete_linear_model(model)
