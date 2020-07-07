import cv2
import os
import numpy as np

from const import img_total_size
from face_id import predict_face


def start_camera_stream(ml_lib, network):
    face_cascade = cv2.CascadeClassifier(
        os.path.abspath("..\\..\\models\\face-detector\\haarcascade_frontalface_default.xml"))
    cap = cv2.VideoCapture(0)

    while True:
        _, img = cap.read()
        faces = face_cascade.detectMultiScale(img, 1.1, 10, minSize=(96, 96))

        if len(faces) > 0:
            (x, y, w, h) = faces[0]
            face = img[y:y + h, x:x + w]
            face = cv2.resize(face, dsize=(48, 48))
            matrix = np.array(face) / 255.0
            matrix = np.reshape(matrix, img_total_size)
            cv2.putText(img, predict_face(ml_lib, network, matrix)[1], (x, y), 0, 1, 255)

        cv2.imshow('img', img)

        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break

    cap.release()
