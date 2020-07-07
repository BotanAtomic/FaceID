package fr.esgi.faceid.controller;

import fr.esgi.faceid.stream.VideoStream;
import javafx.application.Platform;
import javafx.fxml.FXML;
import javafx.scene.image.ImageView;
import javafx.scene.layout.AnchorPane;
import javafx.scene.paint.Color;
import javafx.scene.paint.ImagePattern;
import javafx.scene.shape.Circle;
import javafx.scene.shape.Rectangle;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import java.io.File;
import java.util.UUID;
import java.util.concurrent.atomic.AtomicInteger;

import static fr.esgi.faceid.utils.OpenCV.cropCenter;
import static fr.esgi.faceid.utils.OpenCV.matToImage;

/**
 * Created by Botan on 5/22/2020. 6:50 PM
 **/
public class TrainController {

    private final static int N_COLLECT_IMAGE = 200;
    private final AtomicInteger imageCollected = new AtomicInteger(0);
    private final Rectangle[] rectangles = new Rectangle[60];
    private final String name = UUID.randomUUID().toString().substring(0, 4);
    @FXML
    private Circle faceView;
    @FXML
    private ImageView representation3D;
    @FXML
    private AnchorPane container;

    private Runnable callback;

    public void setCallback(Runnable callback) {
        this.callback = callback;
    }

    @FXML
    private void initialize() {
        File mainPath = new File("data/" + name);
        mainPath.mkdirs();

        VideoStream videoStream = VideoStream.getInstance();

        videoStream.showLandmark(false);
        videoStream.allowMultipleFace(false);
        videoStream.setImagePreprocessing(null);

        videoStream.setMatrixCallback(mat -> faceView.setFill(new ImagePattern(matToImage(cropCenter(mat)))));

        videoStream.setRepresentation3DCallback(mat -> representation3D.setImage(matToImage(mat)));

        videoStream.setFaceCallback(mat -> {
            int current = imageCollected.incrementAndGet();
            double percent = current / (double) N_COLLECT_IMAGE;
            int index = (int) (60 * percent);
            if (current > N_COLLECT_IMAGE) {
                Platform.runLater(() -> callback.run());
                return;
            }

            Imgcodecs.imwrite(new File(mainPath, current + ".jpg").getAbsolutePath(), mat);

            Platform.runLater(() -> {
                if (index < rectangles.length) {
                    Rectangle r = rectangles[index];
                    r.setFill(Color.GREEN);
                }
            });
        });

        for (int i = 0; i < 60; i++) {
            int degree = i * 6;
            double x = 165.2 + (120 * (Math.cos((degree) * (Math.PI / 180.0f))));
            double y = 143 + (120 * (Math.sin((degree) * (Math.PI / 180.0f))));
            Rectangle rectangle = new Rectangle() {{
                setX(x);
                setY(y);
                setHeight(10);
                setWidth(5);
                setRotate(degree - 90);
                setFill(Color.GREY);
            }};
            rectangles[i] = rectangle;
            container.getChildren().add(rectangle);
        }
    }
}
