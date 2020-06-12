package fr.esgi.faceid.ai;

import com.sun.jna.Native;
import fr.esgi.faceid.api.ILinearModel;
import fr.esgi.faceid.core.Main;
import fr.esgi.faceid.entity.User;
import fr.esgi.faceid.math.Normalizer;
import fr.esgi.faceid.stream.VideoStream;
import javafx.application.Platform;
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
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;

import static fr.esgi.faceid.configuration.Configuration.*;
import static fr.esgi.faceid.controller.Controller.LOADING_IMAGE;

/**
 * Created by Botan on 5/28/2020. 11:23 PM
 **/
public class NeuralNetworkManager {

    private final List<User> users = new CopyOnWriteArrayList<>();
    private final NativeImageLoader nativeImageLoader = new NativeImageLoader(IMG_SIZE, IMG_SIZE, IMG_CHANNEL);
    private final ImageView view;

    private final ILinearModel nativeInterface;

    private final AtomicBoolean training = new AtomicBoolean(false);

    public NeuralNetworkManager(ImageView view) {
        this.view = view;
        this.nativeInterface = Native.load(Main.LIB_PATH, ILinearModel.class);
    }

    public synchronized Pair<User, Double> predict(Mat input) {
        if (training.get())
            return null;

        try {
            double[] inputs = nativeImageLoader.asMatrix(input).reshape(IMG_TOTAL_SIZE).div(255).toDoubleVector();
            List<Double> predictions = new ArrayList<>();

            Pair<User, Double> result = users.stream()
                    .filter(user -> user.getNeuralNetwork() != null)
                    .map(user -> {
                        double p = user.getNeuralNetwork().predict(inputs);
                        predictions.add(p);
                        return new Pair<>(user, user.getNeuralNetwork().predict(inputs));
                    })
                    .max((a, b) -> (int) (a.getValue() - b.getValue()))
                    .orElse(null);

            if (result != null)
                result = new Pair<>(result.getKey(),
                        Normalizer.softmax(result.getValue(),
                                predictions.stream().mapToDouble(e -> e).toArray()));

            return result;
        } catch (Exception e) {
            e.printStackTrace();
            return null;
        }
    }

    public void train(Runnable callback) throws Exception {
        if (training.get())
            return;

        training.set(true);
        users.sort(Comparator.comparing(User::getName));
        VideoStream stream = VideoStream.getInstance();
        stream.setFaceCallback(null);
        stream.setMatrixCallback(null);
        stream.setRepresentation3DCallback(null);
        Platform.runLater(() -> view.setImage(LOADING_IMAGE));
        System.out.println("Start new training...");

        int filesSize = users.stream().mapToInt(u -> Objects.requireNonNull(u.getDirectory().listFiles()).length).sum();

        INDArray inputs = Nd4j.create(filesSize, IMG_TOTAL_SIZE);
        final INDArray labels = Nd4j.create(1, filesSize);
        int index = 0;
        for (User user : users) {
            int userIndex = users.indexOf(user);

            for (File image : Objects.requireNonNull(user.getDirectory().listFiles())) {
                if (image.getName().equals("model")) continue;
                INDArray imgArray = nativeImageLoader.asMatrix(image).reshape(IMG_TOTAL_SIZE).div(255);
                inputs.putRow(index, imgArray);
                labels.put(0, index, userIndex);
                imgArray.close();
                index++;
            }
        }

        final INDArray normalizedInputs = inputs.reshape(filesSize * (IMG_TOTAL_SIZE));

        for (User user : users) {
            int userIndex = users.indexOf(user);
            NeuralNetwork neuralNetwork = new NeuralNetwork(IMG_TOTAL_SIZE, nativeInterface);
            INDArray normalizedLabels = labels.dup();

            BooleanIndexing.replaceWhere(normalizedLabels, 0, Conditions.epsNotEquals(userIndex));
            BooleanIndexing.replaceWhere(normalizedLabels, 1, Conditions.epsEquals(userIndex));

            neuralNetwork.train(normalizedInputs, normalizedLabels, (int) normalizedLabels.length(),
                    IMG_TOTAL_SIZE, 10000, 0.01);
            neuralNetwork.save(new File(user.getDirectory(), "model"));

            System.out.println("Training done for " + user.getName());

            user.setNeuralNetwork(neuralNetwork);
            normalizedLabels.close();
        }
        training.set(false);
        callback.run();
        labels.close();
    }

    public List<User> getUsers() {
        return users;
    }

    public void addUser(User user) {
        File modelFile = new File(user.getDirectory(), "model");

        if (modelFile.exists()) {
            user.setNeuralNetwork(NeuralNetwork.from(modelFile, nativeInterface));
        }

        this.users.add(user);
    }

    public boolean isTraining() {
        return training.get();
    }
}
