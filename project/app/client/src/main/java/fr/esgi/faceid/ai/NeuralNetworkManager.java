package fr.esgi.faceid.ai;

import fr.esgi.faceid.controller.Controller;
import fr.esgi.faceid.entity.User;
import fr.esgi.faceid.stream.VideoStream;
import javafx.application.Platform;
import javafx.scene.image.ImageView;
import javafx.util.Pair;
import org.deeplearning4j.ui.api.UIServer;
import org.opencv.core.Mat;

import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * Created by Botan on 5/28/2020. 11:23 PM
 **/
public class NeuralNetworkManager {

    public static UIServer uiServer = UIServer.getInstance();


    private final List<User> users = new CopyOnWriteArrayList<>();
    private final Controller controller;
    private final AtomicBoolean training = new AtomicBoolean(false);
    private final Map<String, Class<? extends NeuralNetwork>> availableNeuralNetwork = new HashMap<>() {{
        put("linear", LinearNeuralNetwork.class);
        put("mlp", MLPNetwork.class);
        put("dl4j", DL4JNetwork.class);
        put("dl4j_cnn", DL4JCNNNetwork.class);
    }};
    private NeuralNetwork neuralNetwork;


    public NeuralNetworkManager(Controller controller) {
        this.controller = controller;
    }

    public void setNeuralNetwork(String neuralNetwork) {
        try {
            this.neuralNetwork = (NeuralNetwork) availableNeuralNetwork.getOrDefault(neuralNetwork, availableNeuralNetwork.get("linear"))
                    .getConstructors()[0].newInstance(users);
            System.out.println("Set neural network: " + neuralNetwork);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public synchronized Pair<User, Integer> predict(Mat input) {
        if (training.get() || neuralNetwork == null)
            return null;

        try {
            return neuralNetwork.predict(input);
        } catch (Exception e) {
            e.printStackTrace();
            return null;
        }
    }

    public void train() throws Exception {
        if (training.get())
            return;

        training.set(true);
        users.sort(Comparator.comparing(User::getName));
        VideoStream stream = VideoStream.getInstance();
        if (stream != null) {
            stream.setFaceCallback(null);
            stream.setMatrixCallback(null);
            stream.setRepresentation3DCallback(null);
        }
        Thread.sleep(500);
        Platform.runLater(controller::setLoadingImage);
        System.out.println("Start new training...");
        neuralNetwork.train();
        training.set(false);
        System.out.println("Training done");
        Platform.runLater(controller::setDoneImage);
    }

    public List<User> getUsers() {
        return users;
    }

    public void addUser(User user) {
        this.users.add(user);
        if (neuralNetwork != null)
            neuralNetwork.addUser(user);
    }

    public boolean isTraining() {
        return training.get();
    }

    public void invalidateNetwork() {
        if (neuralNetwork != null)
            this.neuralNetwork.invalidate();
    }
}
