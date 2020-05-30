package fr.esgi.faceid.controller;

import fr.esgi.faceid.ai.NeuralNetworkManager;
import fr.esgi.faceid.entity.User;
import fr.esgi.faceid.stream.VideoStream;
import fr.esgi.faceid.utils.OpenCV;
import javafx.fxml.FXML;
import javafx.fxml.FXMLLoader;
import javafx.scene.Scene;
import javafx.scene.control.Label;
import javafx.scene.image.ImageView;
import javafx.scene.layout.AnchorPane;
import javafx.scene.layout.HBox;
import javafx.scene.layout.VBox;
import javafx.scene.paint.ImagePattern;
import javafx.scene.shape.Circle;
import javafx.stage.Stage;
import javafx.stage.StageStyle;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import java.io.File;
import java.util.Objects;
import java.util.concurrent.Executor;
import java.util.concurrent.Executors;

import static fr.esgi.faceid.utils.OpenCV.matToImage;
import static fr.esgi.faceid.utils.UI.implementAvatarContextMenu;

/**
 * Created by Botan on 5/22/2020. 6:50 PM
 **/
public class Controller {

    private final Executor executor = Executors.newSingleThreadExecutor();

    @FXML
    private AnchorPane root;
    @FXML
    private VBox datasetView;
    @FXML
    private ImageView webcamView;
    private boolean trainOpened = false;
    private NeuralNetworkManager neuralNetworkManager;


    @FXML
    private void initialize() {
        neuralNetworkManager = new NeuralNetworkManager(webcamView);

        initCamera();
        loadUsers();

        webcamView.fitWidthProperty().bind(root.widthProperty());
        webcamView.fitHeightProperty().bind(root.heightProperty());
    }

    private void initCamera() {
        VideoStream stream = VideoStream.getInstance();
        stream.allowMultipleFace(true);
        stream.setImagePreprocessing((matrix, face, coordinates) -> {
            User prediction = neuralNetworkManager.predict(face);
            if (prediction == null) return;
            Size textWidth = Imgproc.getTextSize(prediction.getName(), 1, 2, 1, new int[]{1});
            Imgproc.putText(matrix, prediction.getName(), OpenCV.middlePoint(
                    new Point(coordinates.x - textWidth.width / 2, coordinates.y + textWidth.height),
                    new Point((coordinates.x + coordinates.width) - textWidth.width / 2, coordinates.y + textWidth.height)
            ), 1, 2, new Scalar(255));
        });
        stream.setMatrixCallback(mat -> webcamView.setImage(matToImage(mat)));
        stream.showLandmark(true);
    }

    private void loadUsers() {
        for (File file : Objects.requireNonNull(new File("data").listFiles(File::isDirectory))) {
            addUser(new User(file));
        }
    }

    private void renameUser(User user, String newName) {
        datasetView.getChildren()
                .stream()
                .filter(node -> user.getName().equals(node.getAccessibleRoleDescription()))
                .findAny()
                .ifPresent(node -> node.setAccessibleRoleDescription(newName));
        File newDirectory = new File("data/" + newName);
        user.getDirectory().renameTo(newDirectory);
        user.setName(newName);
        user.setDirectory(newDirectory);
    }

    private void removeUser(User user) {
        datasetView.getChildren().removeIf(node -> user.getName().equals(node.getAccessibleRoleDescription()));
        neuralNetworkManager.getUsers().remove(user);
        try {
            for (File file : Objects.requireNonNull(user.getDirectory().listFiles()))
                file.delete();

            user.getDirectory().delete();
            if (neuralNetworkManager.getUsers().size() > 1) {
                train();
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private void addUser(User user) {
        if (neuralNetworkManager.getUsers().stream().anyMatch(u -> u.getName().equals(user.getName())))
            return;

        try {
            HBox root = FXMLLoader.load(getClass().getResource("/avatar.fxml"));
            ((Circle) root.lookup("#image")).setFill(new ImagePattern(user.randomImage()));
            ((Label) root.lookup("#name")).setText(user.getName());
            ((Label) root.lookup("#count")).setText(String.format("%s images", user.countImages()));
            root.setAccessibleRoleDescription(user.getName());
            implementAvatarContextMenu(root, () -> removeUser(user), (newName) -> {
                renameUser(user, newName);
                ((Label) root.lookup("#name")).setText(user.getName());
            });
            neuralNetworkManager.addUser(user);
            datasetView.getChildren().add(root);
        } catch (Exception e) {
            e.printStackTrace();
        }

    }

    @FXML
    private void train() {
        executor.execute(() -> {
            try {
                neuralNetworkManager.train(this::initCamera);
            } catch (Exception e) {
                e.printStackTrace();
            }
        });

    }

    @FXML
    private void addFace() {
        if (trainOpened)
            return;

        Runnable callback = () -> {
            initCamera();
            loadUsers();
            trainOpened = false;
        };

        try {
            trainOpened = true;
            Stage stage = new Stage();
            FXMLLoader root = new FXMLLoader(getClass().getResource("/train.fxml"));
            stage.setScene(new Scene(root.load()));
            ((TrainController) root.getController()).setCallback(() -> {
                callback.run();
                stage.close();

                if (neuralNetworkManager.getUsers().size() > 1) {
                    train();
                }
            });
            stage.initStyle(StageStyle.UNDECORATED);
            stage.show();

            stage.setOnCloseRequest(e -> callback.run());
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
