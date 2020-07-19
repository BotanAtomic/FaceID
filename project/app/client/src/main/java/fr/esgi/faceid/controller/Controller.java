package fr.esgi.faceid.controller;

import fr.esgi.faceid.ai.NeuralNetworkManager;
import fr.esgi.faceid.entity.User;
import fr.esgi.faceid.stream.VideoStream;
import fr.esgi.faceid.utils.OpenCV;
import javafx.application.Platform;
import javafx.fxml.FXML;
import javafx.fxml.FXMLLoader;
import javafx.geometry.Insets;
import javafx.scene.Scene;
import javafx.scene.control.ChoiceBox;
import javafx.scene.control.Label;
import javafx.scene.image.Image;
import javafx.scene.image.ImageView;
import javafx.scene.input.Dragboard;
import javafx.scene.input.TransferMode;
import javafx.scene.layout.*;
import javafx.scene.paint.Color;
import javafx.scene.paint.ImagePattern;
import javafx.scene.shape.Circle;
import javafx.stage.Stage;
import javafx.stage.StageStyle;
import javafx.util.Pair;
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
import static fr.esgi.faceid.utils.UI.implementImageContextMenu;

/**
 * Created by Botan on 5/22/2020. 6:50 PM
 **/
public class Controller {

    private static final Image LOADING_IMAGE = new Image(NeuralNetworkManager.class.getResourceAsStream("/loading.gif"));
    private static final Image DONE_IMAGE = new Image(NeuralNetworkManager.class.getResourceAsStream("/done.gif"));

    private final Executor executor = Executors.newSingleThreadExecutor();

    @FXML
    private AnchorPane root;
    @FXML
    private StackPane rootImg;
    @FXML
    private VBox datasetView, dragAndDropRoot;
    @FXML
    private ImageView webcamView;

    @FXML
    private ChoiceBox<String> aiChoice;

    private boolean trainOpened = false;
    private NeuralNetworkManager neuralNetworkManager;

    private boolean enableDragAndDrop = true;


    @FXML
    private void initialize() {
        aiChoice.getItems().addAll("Linear", "MLP", "DL4J", "DL4J_CNN");
        aiChoice.setValue("Linear");
        neuralNetworkManager = new NeuralNetworkManager(this);

        dragAndDrop(true);
        loadUsers();

        implementImageContextMenu(webcamView, () -> {
            if (VideoStream.getInstance() != null)
                VideoStream.getInstance().close();
            dragAndDrop(true);
        });

        neuralNetworkManager.setNeuralNetwork("linear");

        aiChoice.getSelectionModel().selectedItemProperty().addListener((a, b, c) -> {
            dragAndDrop(false);
            setLoadingImage();
            executor.execute(() -> {
                neuralNetworkManager.setNeuralNetwork(c.toLowerCase());
                Platform.runLater(this::setDoneImage);
                dragAndDrop(true);
            });
        });

        root.setOnDragOver(event -> {
            if (enableDragAndDrop) {
                rootImg.setBorder(new Border(new BorderStroke(Color.GREEN, BorderStrokeStyle.DASHED, new CornerRadii(5), new BorderWidths(3),
                        new Insets(10, 10, 10, 10))));
                event.acceptTransferModes(TransferMode.COPY_OR_MOVE);
            }
            event.consume();
        });

        root.setOnDragDropped(event -> {
            if (enableDragAndDrop) {
                Dragboard db = event.getDragboard();
                if (db.hasFiles()) {
                    dragAndDrop(false);
                    if (VideoStream.getInstance() != null)
                        VideoStream.getInstance().close();
                    new VideoStream(db.getFiles().get(0))
                            .allowMultipleFace(false);
                    initCamera();
                }
            }
            event.setDropCompleted(enableDragAndDrop);
            event.consume();
        });

        root.setOnDragExited(e -> {
            if (enableDragAndDrop) {
                rootImg.setBorder(new Border(new BorderStroke(Color.GRAY, BorderStrokeStyle.DASHED, new CornerRadii(5), new BorderWidths(3),
                        new Insets(10, 10, 10, 10))));
            }
        });
        setDoneImage();
    }

    private void initCamera() {
        setLoadingImage();

        VideoStream stream = VideoStream.getInstance();
        stream.allowMultipleFace(true);
        stream.setImagePreprocessing((matrix, face, coordinates) -> {
            Pair<User, Integer> prediction = neuralNetworkManager.predict(face);
            if (prediction == null) return;
            String text;
            if (prediction.getValue() > 0) {
                text = String.format("%s (%s)", prediction.getKey().getName(), prediction.getValue() + "%");
            } else {
                text = String.format("%s", prediction.getKey().getName());
            }
            Size textWidth = Imgproc.getTextSize(text, 1, 1, 1, new int[]{1});
            Imgproc.putText(matrix, text, OpenCV.middlePoint(
                    new Point(coordinates.x - textWidth.width / 2, coordinates.y + textWidth.height),
                    new Point((coordinates.x + coordinates.width) - textWidth.width / 2, coordinates.y + textWidth.height)
            ), 1, 1, new Scalar(255));
        });
        webcamView.fitWidthProperty().bind(rootImg.widthProperty());
        webcamView.fitHeightProperty().bind(rootImg.heightProperty());
        stream.setMatrixCallback(mat -> webcamView.setImage(matToImage(mat)));
        stream.setFaceCallback(null);
        stream.showLandmark(true);
        stream.setOnEndStream(() -> dragAndDrop(true));
    }

    public void setLoadingImage() {
        webcamView.setVisible(true);
        webcamView.fitWidthProperty().bind(rootImg.widthProperty());
        webcamView.fitHeightProperty().bind(rootImg.heightProperty());
        webcamView.setImage(LOADING_IMAGE);
    }

    public void setDoneImage() {
        webcamView.fitWidthProperty().unbind();
        webcamView.fitHeightProperty().unbind();
    }

    private void dragAndDrop(boolean enable) {
        if(!enable) {
            rootImg.setBorder(Border.EMPTY);
            webcamView.setVisible(true);
            dragAndDropRoot.setVisible(false);
        } else {
            webcamView.setVisible(false);
            dragAndDropRoot.setVisible(true);
            rootImg.setBorder(new Border(new BorderStroke(Color.GRAY, BorderStrokeStyle.DASHED, new CornerRadii(5), new BorderWidths(3),
                    new Insets(10, 10, 10, 10))));
        }
        enableDragAndDrop = enable;
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

        new File("models/linear", user.getName() + ".model")
                .renameTo(new File("models/linear", newName + ".model"));

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

            new File("models/linear", user.getName() + ".model").delete();
            neuralNetworkManager.invalidateNetwork();
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
        if (neuralNetworkManager.isTraining())
            return;

        executor.execute(() -> {
            try {
                neuralNetworkManager.train();
            } catch (Exception e) {
                e.printStackTrace();
            }
        });

    }

    @FXML
    private void addFace() {
        if (trainOpened)
            return;

        if(VideoStream.getInstance() != null)
            VideoStream.getInstance().close();

        Runnable callback = () -> {
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
            });
            stage.initStyle(StageStyle.UNDECORATED);
            stage.show();

            stage.setOnCloseRequest(e -> callback.run());
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    @FXML
    private void enableWebcam() {
        dragAndDrop(false);
        if (VideoStream.getInstance() != null)
            VideoStream.getInstance().close();
        new VideoStream();
        initCamera();
    }
}
