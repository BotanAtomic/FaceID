package fr.esgi.faceid.ai;

import com.sun.jna.Native;
import fr.esgi.faceid.api.ILinearModel;
import fr.esgi.faceid.configuration.Configuration;
import fr.esgi.faceid.core.Main;
import fr.esgi.faceid.entity.User;
import fr.esgi.faceid.stream.VideoStream;
import javafx.application.Platform;
import javafx.scene.image.Image;
import javafx.scene.image.ImageView;
import javafx.util.Pair;
import org.datavec.image.loader.NativeImageLoader;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.BooleanIndexing;
import org.nd4j.linalg.indexing.conditions.Conditions;
import org.opencv.core.Mat;

import java.io.File;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Objects;
import java.util.concurrent.atomic.AtomicBoolean;

import static fr.esgi.faceid.configuration.Configuration.*;

/**
 * Created by Botan on 5/28/2020. 11:23 PM
 **/
public class NeuralNetworkManager {

    private final List<User> users = new ArrayList<>();
    private final NativeImageLoader nativeImageLoader = new NativeImageLoader(IMG_SIZE, IMG_SIZE, IMG_CHANNEL);
    private ImageView view;

    private ILinearModel nativeInterface;

    private AtomicBoolean training = new AtomicBoolean(false);

    public NeuralNetworkManager(ImageView view) {
        this.view = view;
        this.nativeInterface = Native.load(Main.LIB_PATH, ILinearModel.class);
    }

    public User predict(Mat input) {
        if (training.get())
            return null;
        try {
            double[] inputs = nativeImageLoader.asMatrix(input).reshape(IMG_TOTAL_SIZE).div(255).toDoubleVector();
            Pair<User, Double> results = users.stream()
                    .filter(user -> user.getNeuralNetwork() != null)
                    .map(user -> new Pair<>(user, user.getNeuralNetwork().predict(inputs)))
                    .max((a, b) -> (int) (a.getValue() - b.getValue()))
                    .orElse(null);

            return results == null ? null : results.getKey();
        } catch (Exception e) {
            e.printStackTrace();
            return null;
        }
    }

    public void train(Runnable callback) throws Exception {
        training.set(true);
        users.sort(Comparator.comparing(User::getName));
        VideoStream stream = VideoStream.getInstance();
        stream.setFaceCallback(null);
        stream.setMatrixCallback(null);
        stream.setRepresentation3DCallback(null);
        Platform.runLater(() -> view.setImage(new Image(NeuralNetworkManager.class.getResourceAsStream("/loading.gif"))));

        int filesSize = users.stream().mapToInt(u -> Objects.requireNonNull(u.getDirectory().listFiles()).length).sum();

        INDArray inputs = Nd4j.create(filesSize, IMG_TOTAL_SIZE);
        INDArray labels = Nd4j.create(1, filesSize);
        int index = 0;
        for (User user : users) {
            int userIndex = users.indexOf(user);

            for (File image : Objects.requireNonNull(user.getDirectory().listFiles())) {
                INDArray imgArray = nativeImageLoader.asMatrix(image).reshape(IMG_TOTAL_SIZE).div(255);
                inputs.putRow(index, imgArray);
                labels.put(0, index, userIndex);
                index++;
            }
        }

        for (User user : users) {
            int userIndex = users.indexOf(user);
            NeuralNetwork neuralNetwork = new NeuralNetwork(IMG_TOTAL_SIZE, nativeInterface);
            INDArray normalizedLabels = labels.dup();

            BooleanIndexing.replaceWhere(normalizedLabels, -1, Conditions.epsNotEquals(userIndex));
            BooleanIndexing.replaceWhere(normalizedLabels, 1, Conditions.epsEquals(userIndex));

            neuralNetwork.train(inputs.reshape(filesSize * (IMG_TOTAL_SIZE)), normalizedLabels, (int) normalizedLabels.length(),
                    IMG_TOTAL_SIZE, 10000, 0.01);
            neuralNetwork.save(new File(user.getDirectory(), "model"));

            System.out.println("Training done for " + user.getName());

            user.setNeuralNetwork(neuralNetwork);
            normalizedLabels.close();
        }

        training.set(false);
        callback.run();
        inputs.close();
        labels.close();
    }

    public List<User> getUsers() {
        return users;
    }

    public void addUser(User user) {
        File modelFile = new File(user.getDirectory(), "model");

        if(modelFile.exists()) {
            user.setNeuralNetwork(NeuralNetwork.from(modelFile, nativeInterface));
        }

        this.users.add(user);
    }
}
