package fr.esgi.faceid.stream;

/**
 * Created by Botan on 5/22/2020. 6:52 PM
 **/

import fr.esgi.faceid.utils.OpenCV;
import javafx.scene.image.Image;
import javafx.scene.image.ImageView;
import org.opencv.core.*;
import org.opencv.face.Face;
import org.opencv.face.Facemark;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;
import org.opencv.objdetect.Objdetect;
import org.opencv.videoio.VideoCapture;

import java.io.ByteArrayInputStream;
import java.util.ArrayList;

import static fr.esgi.faceid.utils.OpenCV.FACIAL_LINE;
import static fr.esgi.faceid.utils.OpenCV.FACIAL_POINTS;
import static org.opencv.imgcodecs.Imgcodecs.imencode;

public class VideoStream extends Thread {

    private final static Size faceSize = new Size(100, 100);

    private final ImageView canvas;
    private final VideoCapture videoCapture;
    private final CascadeClassifier faceCascade;
    private final Facemark facemark;

    private OnReceiveFace callback;


    public VideoStream(ImageView canvas) {
        this.canvas = canvas;
        System.out.println(canvas);
        this.videoCapture = new VideoCapture();
        this.faceCascade = new CascadeClassifier("../../models/face-detector/haarcascade_frontalface_alt.xml");
        this.facemark = Face.createFacemarkLBF();
        this.facemark.loadModel("../../models/landmark/lbfmodel.yaml");
    }

    public void setCallback(OnReceiveFace callback) {
        this.callback = callback;
    }

    private void writeToCanvas(Mat matrix) {
        MatOfByte buffer = new MatOfByte();
        imencode(".jpg", matrix, buffer);

        Image image = new Image(new ByteArrayInputStream(buffer.toArray()));
        canvas.setImage(image);
        buffer.release();
    }

    private Point[] normalizeFacemark(Point[] points) {
        points[19] = OpenCV.middlePoint(points[19], points[20]);
        points[24] = OpenCV.middlePoint(points[24], points[23]);
        return points;
    }

    @Override
    public void run() {
        videoCapture.open(0);

        Mat matrix = new Mat();
        while (videoCapture.isOpened()) {
            videoCapture.read(matrix);

            if (!matrix.empty()) {
                MatOfRect faces = new MatOfRect();
                faceCascade.detectMultiScale(matrix, faces, 1.1, 2, Objdetect.CASCADE_SCALE_IMAGE,
                        faceSize, new Size());

                Rect[] facesArray = faces.toArray();

                for (Rect faceRect : facesArray) {
                    Mat face = new Mat(matrix, faceRect);


                    if (callback != null)
                        callback.onReceive(face);

                    face.release();
                }

                ArrayList<MatOfPoint2f> landmarks = new ArrayList<>();
                facemark.fit(matrix, faces, landmarks);
                for (MatOfPoint2f lm : landmarks) {
                    Point[] points = normalizeFacemark(lm.toArray());
                    for (int i : FACIAL_POINTS) {
                        Imgproc.circle(matrix, points[i], 2, new Scalar(255, 255, 255), 2);
                    }

                    for (int[] lines : FACIAL_LINE) {
                        Imgproc.line(matrix, points[lines[0]], points[lines[1]], new Scalar(255, 255, 255), 1);
                    }

                }
            }

            writeToCanvas(matrix);
            matrix.release();
        }
    }

    public interface OnReceiveFace {
        void onReceive(Mat mat);
    }

}

