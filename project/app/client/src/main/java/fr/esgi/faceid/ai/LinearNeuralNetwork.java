package fr.esgi.faceid.ai;

import com.sun.jna.Native;
import fr.esgi.faceid.api.ILinearModel;
import fr.esgi.faceid.core.Main;
import fr.esgi.faceid.entity.User;
import javafx.util.Pair;
import org.datavec.image.loader.NativeImageLoader;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.BooleanIndexing;
import org.nd4j.linalg.indexing.conditions.Conditions;
import org.opencv.core.Mat;

import java.io.File;
import java.util.List;
import java.util.Objects;


/**
 * Created by Botan on 5/28/2020. 10:43 PM
 **/
public class LinearNeuralNetwork implements NeuralNetwork {

    private final static int IMG_SIZE = 48;

    private final static int IMG_CHANNEL = 3;

    public final static int IMG_TOTAL_SIZE = IMG_SIZE * IMG_SIZE * IMG_CHANNEL;

    public final NativeImageLoader NATIVE_IMAGE_LOADER = new NativeImageLoader(IMG_SIZE, IMG_SIZE, IMG_CHANNEL);

    private final int modelSize = IMG_TOTAL_SIZE;

    private final ILinearModel nativeInterface;

    private final List<User> users;

    public LinearNeuralNetwork(List<User> users) {
        this.nativeInterface = Native.load(Main.LIB_PATH, ILinearModel.class);
        this.users = users;
        users.forEach(this::addUser);
    }

    public User predict(Mat input) throws Exception {
        if (users.isEmpty()) return null;

        double[] data = NATIVE_IMAGE_LOADER.asMatrix(input).reshape(IMG_TOTAL_SIZE).div(255).toDoubleVector();

        try {
            Pair<User, Double> result = users.stream()
                    .filter(user -> user.getNeuralNetwork() != null)
                    .map(user -> {
                        double p = nativeInterface.predictRegressionModel(user.getNeuralNetwork(), data, data.length);
                        return new Pair<>(user, p);
                    })
                    .max((a, b) -> (int) (a.getValue() - b.getValue()))
                    .orElse(null);

            if (result != null)
                return result.getKey();

            return null;
        } catch (Exception e) {
            return null;
        }
    }

    @Override
    public void train() throws Exception {
        int filesSize = users.stream().mapToInt(u -> Objects.requireNonNull(u.getDirectory().listFiles()).length).sum();

        INDArray inputs = Nd4j.create(filesSize, IMG_TOTAL_SIZE);
        final INDArray labels = Nd4j.create(1, filesSize);
        int index = 0;
        for (User user : users) {
            if (user.getNeuralNetwork() != null) {
                nativeInterface.deleteModel(user.getNeuralNetwork());
            }
            user.setNeuralNetwork(nativeInterface.createModel(IMG_TOTAL_SIZE));

            int userIndex = users.indexOf(user);

            for (File image : Objects.requireNonNull(user.getDirectory().listFiles())) {
                if (image.getName().equals("model")) continue;
                INDArray imgArray = NATIVE_IMAGE_LOADER.asMatrix(image).reshape(IMG_TOTAL_SIZE).div(255);
                inputs.putRow(index, imgArray);
                labels.put(0, index, userIndex);
                imgArray.close();
                index++;
            }
        }

        final INDArray normalizedInputs = inputs.reshape(filesSize * (IMG_TOTAL_SIZE));

        for (User user : users) {
            int userIndex = users.indexOf(user);
            INDArray normalizedLabels = labels.dup();

            BooleanIndexing.replaceWhere(normalizedLabels, 0, Conditions.epsNotEquals(userIndex));
            BooleanIndexing.replaceWhere(normalizedLabels, 1, Conditions.epsEquals(userIndex));

            train(user, normalizedInputs, normalizedLabels, (int) normalizedLabels.length(),
                    IMG_TOTAL_SIZE, 20000, 0.1);
            save(user, new File("models/linear/", user.getName() + ".model"));

            System.out.println("Training done for " + user.getName());

            normalizedLabels.close();
        }
        labels.close();
    }

    @Override
    public void addUser(User u) {
        File savedNeuralNetwork = new File("models/linear/", u.getName() + ".model");
        if (savedNeuralNetwork.exists()) {
            System.out.println("Restore neural network for user: " + u.getName());
            u.setNeuralNetwork(nativeInterface.loadModel(savedNeuralNetwork.getAbsolutePath()));
        } else {
            u.setNeuralNetwork(nativeInterface.createModel(modelSize));
        }
    }

    @Override
    public void invalidate() {
    }

    public void train(User user, INDArray inputs, INDArray labels, int length, int dataSize, int epoch, double alpha) {
        nativeInterface.trainModel(user.getNeuralNetwork(), inputs.toDoubleVector(), labels.toDoubleVector(), length, dataSize, epoch, alpha);
    }

    public void save(User user, File file) {
        nativeInterface.saveModel(user.getNeuralNetwork(), modelSize, file.getAbsolutePath());
    }
}
