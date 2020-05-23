package fr.esgi.faceid.controller;

import fr.esgi.faceid.stream.VideoStream;
import javafx.fxml.FXML;
import javafx.scene.canvas.Canvas;
import javafx.scene.image.ImageView;
import javafx.scene.layout.AnchorPane;
import javafx.scene.layout.VBox;

/**
 * Created by Botan on 5/22/2020. 6:50 PM
 **/
public class Controller {

    @FXML
    private AnchorPane root;

    @FXML
    private VBox datasetView;

    @FXML
    private ImageView webcamView;

    private VideoStream videoStream;

    @FXML
    private void initialize() {
        videoStream = new VideoStream(webcamView);
        videoStream.start();

        webcamView.fitWidthProperty().bind(root.widthProperty());
        webcamView.fitHeightProperty().bind(root.heightProperty());
    }
}
