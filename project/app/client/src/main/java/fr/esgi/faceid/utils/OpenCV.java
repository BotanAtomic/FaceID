package fr.esgi.faceid.utils;


import javafx.scene.image.Image;
import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
import org.opencv.core.Point;
import org.opencv.core.Rect;

import java.io.ByteArrayInputStream;

import static org.opencv.imgcodecs.Imgcodecs.imencode;

/**
 * Created by Botan on 5/23/2020. 11:22 AM
 **/
public class OpenCV {

    public final static int[] FACIAL_POINTS = {19, 24, 36, 39, 42, 45, 35, 33, 31, 54, 48, 12, 4, 8};
    public final static int[][] FACIAL_LINE = {
            {19, 24},
            {36, 39},
            {42, 45},
            {39, 42},
            {36, 19},
            {19, 39},
            {42, 24},
            {24, 45},
            {45, 12},
            {36, 4},
            {8, 4},
            {12, 8},
            {54, 8},
            {48, 8},
            {48, 4},
            {12, 54},
            {48, 33},
            {54, 33},
            {12, 35},
            {4, 31},
            {33, 31},
            {33, 35},
            {33, 36},
            {33, 39},
            {33, 42},
            {33, 45},
    };

    public static Point middlePoint(Point p, Point p2) {
        return new Point((p.x + p2.x) / 2, (p.y + p2.y) / 2);
    }

    public static Image matToImage(Mat matrix) {
        MatOfByte buffer = new MatOfByte();
        imencode(".jpg", matrix, buffer);

        Image image = new Image(new ByteArrayInputStream(buffer.toArray()));
        buffer.release();
        return image;
    }

    public static Mat cropCenter(Mat src) {
        int width = src.width();
        int height = src.height();

        return new Mat(src, new Rect(width / 4, 0, width / 2, height));
    }
}
