package fr.esgi.faceid.stream;

/**
 * Created by Botan on 5/22/2020. 6:52 PM
 **/

import fr.esgi.faceid.utils.OpenCV;
import org.opencv.core.*;
import org.opencv.face.Face;
import org.opencv.face.Facemark;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;
import org.opencv.objdetect.Objdetect;
import org.opencv.videoio.VideoCapture;

import java.io.File;
import java.util.ArrayList;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.Stream;

import static fr.esgi.faceid.utils.OpenCV.FACIAL_LINE;
import static fr.esgi.faceid.utils.OpenCV.FACIAL_POINTS;
import static org.opencv.core.CvType.CV_8UC3;

public class VideoStream extends Thread {

    private static VideoStream instance;

    private final static Size minFaceSize = new Size(80, 80);

    private final VideoCapture videoCapture;
    private final CascadeClassifier faceCascade = new CascadeClassifier("../../models/face-detector/haarcascade_frontalface_default.xml");
    private final Facemark facemark;

    private final AtomicBoolean allowMultipleFace = new AtomicBoolean(true);
    private final AtomicBoolean showLandmark = new AtomicBoolean(true);

    private OnReceiveFace matrixCallback, faceCallback, representation3DCallback;
    private ImagePreprocessing imagePreprocessing;

    private final AtomicBoolean running = new AtomicBoolean(true);
    private final File videoFile;


    public VideoStream(File file) {
        this.videoCapture = new VideoCapture();
        this.facemark = Face.createFacemarkLBF();
        this.facemark.loadModel("../../models/landmark/lbfmodel.yaml");
        this.videoFile = file;
        instance = this;
        System.out.println("Set video stream with videoFile: " + (videoFile != null ? videoFile.getAbsolutePath() : "null"));
        start();
    }

    public VideoStream() {
        this(null);
    }


    public static VideoStream getInstance() {
        return instance;
    }

    private Point[] normalizeFacemark(Point[] points) {
        points[19] = OpenCV.middlePoint(points[19], points[20]);
        points[24] = OpenCV.middlePoint(points[24], points[23]);
        return points;
    }

    @Override
    public void run() {
        boolean opened = false;
        if(videoFile != null)
            opened = videoCapture.open(videoFile.getAbsolutePath());
        else
            opened = videoCapture.open(0);

        System.out.println("Opened : " + opened);

        Mat matrix = new Mat();
        while (videoCapture.isOpened() && running.get()) {
            videoCapture.read(matrix);

            if (!matrix.empty()) {
                MatOfRect faces = new MatOfRect();

                faceCascade.detectMultiScale(matrix, faces, 1.1, 10, Objdetect.CASCADE_FIND_BIGGEST_OBJECT,
                        minFaceSize, new Size());


                Rect[] facesArray = faces.toArray();

                int i = 0;

                for (Rect faceRect : facesArray) {
                    if (i > 0 && !allowMultipleFace.get())
                        break;

                    Mat face = new Mat(matrix, faceRect);

                    if (faceCallback != null)
                        faceCallback.onReceive(face);

                    if (imagePreprocessing != null)
                        imagePreprocessing.process(matrix, face, faceRect);

                    face.release();
                    i++;
                }

                i = 0;

                ArrayList<MatOfPoint2f> landmarks = new ArrayList<>();
                facemark.fit(matrix, faces, landmarks);

                for (MatOfPoint2f lm : landmarks) {
                    if (i > 0 && !allowMultipleFace.get())
                        break;

                    Point[] points = normalizeFacemark(lm.toArray());

                    double xOffset = points[0].x - 10;
                    double yOffset = Math.max(points[19].y, points[24].y) - 10;

                    Mat representation3D = new Mat((facesArray[0].height) + 10,
                            (facesArray[0].width * 2) + 10, CV_8UC3, new Scalar(0));

                    if (showLandmark.get()) {
                        for (int facialPointIndex : FACIAL_POINTS)
                            Imgproc.circle(matrix, points[facialPointIndex], 2, new Scalar(255, 255, 255), 2);

                    }

                    for (int[] lines : FACIAL_LINE) {
                        Point a = points[lines[0]];
                        Point b = points[lines[1]];

                        if (showLandmark.get())
                            Imgproc.line(matrix, a, b, new Scalar(255, 255, 255), 1);
                        else
                            Imgproc.line(representation3D,
                                    new Point((a.x - xOffset) + facesArray[0].width + 20, (a.y - yOffset)),
                                    new Point((b.x - xOffset) + facesArray[0].width + 20, b.y - yOffset),
                                    new Scalar(255, 255, 255), 2);
                    }

                    if (!showLandmark.get()) {
                        for (Point point : points) {
                            Imgproc.circle(representation3D, new Point(point.x - xOffset, point.y - yOffset),
                                    2, new Scalar(255, 255, 255), 2);
                        }
                    }


                    if (representation3DCallback != null)
                        representation3DCallback.onReceive(representation3D);

                    representation3D.release();
                    i++;
                }

                if (matrixCallback != null)
                    matrixCallback.onReceive(matrix);

                matrix.release();
            }
        }

        videoCapture.release();
    }

    public void setMatrixCallback(OnReceiveFace onReceiveFace) {
        this.matrixCallback = onReceiveFace;
    }

    public void setFaceCallback(OnReceiveFace onReceiveFace) {
        this.faceCallback = onReceiveFace;
    }

    public void setRepresentation3DCallback(OnReceiveFace onReceiveFace) {
        this.representation3DCallback = onReceiveFace;
    }

    public void showLandmark(boolean value) {
        this.showLandmark.set(value);
    }

    public void allowMultipleFace(boolean value) {
        this.allowMultipleFace.set(value);
    }

    public void setImagePreprocessing(ImagePreprocessing preprocessing) {
        this.imagePreprocessing = preprocessing;
    }

    public void close() {
        this.running.set(false);
    }

    public interface OnReceiveFace {
        void onReceive(Mat mat);
    }

    public interface ImagePreprocessing {
        void process(Mat matrix, Mat face, Rect coordinates);
    }

}

