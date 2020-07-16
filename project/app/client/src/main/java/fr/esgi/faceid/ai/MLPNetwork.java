package fr.esgi.faceid.ai;

import com.sun.jna.Native;
import com.sun.jna.Pointer;
import fr.esgi.faceid.api.IMultiLayerNetwork;
import fr.esgi.faceid.core.Main;
import fr.esgi.faceid.entity.User;
import org.datavec.image.loader.NativeImageLoader;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.opencv.core.Mat;

import java.io.File;
import java.util.List;
import java.util.Objects;

/**
 * Created by Botan on 5/28/2020. 10:43 PM
 **/
public class MLPNetwork implements NeuralNetwork {

    private final static int IMG_SIZE = 28;

    private final static int IMG_CHANNEL = 1;

    public final static int IMG_TOTAL_SIZE = IMG_SIZE * IMG_SIZE * IMG_CHANNEL;

    public final NativeImageLoader NATIVE_IMAGE_LOADER = new NativeImageLoader(IMG_SIZE, IMG_SIZE, IMG_CHANNEL);

    private final IMultiLayerNetwork nativeInterface;

    private final List<User> users;

    private Pointer model;

    public MLPNetwork(List<User> users) {
        this.nativeInterface = Native.load(Main.MLP_LIB_PATH, IMultiLayerNetwork.class);
        this.users = users;

        createOrLoadModel(true);
    }

    private void createOrLoadModel(boolean load) {
        File savedNeuralNetwork = new File("models/mlp/neuralNetwork.model");

        if (!load && model != null) {
            nativeInterface.deleteModel(model);
        }

        if (!savedNeuralNetwork.exists() || !load) {
            this.model = nativeInterface.createModel(IMG_TOTAL_SIZE);
            nativeInterface.addLayer(model, 128, "activation=relu");
            nativeInterface.addLayer(model, 128, "activation=sigmoid");
            nativeInterface.addLayer(model, users.size(), "activation=sigmoid");
        } else {
            this.model = nativeInterface.loadModel(savedNeuralNetwork.getAbsolutePath());
            System.out.println("Restore MLP model");
        }
    }

    public User predict(Mat input) throws Exception {
        if(model == null) return null;

        double[] inputs = NATIVE_IMAGE_LOADER.asMatrix(input).reshape(IMG_TOTAL_SIZE).div(255).toDoubleVector();
        double[] result = nativeInterface.predict(model, inputs).getDoubleArray(0, users.size());

        return users.get((int) Nd4j.argMax(Nd4j.create(result), 0).toDoubleVector()[0]);
    }

    @Override
    public void train() throws Exception {
        System.out.println("Train MLP");

        createOrLoadModel(false);

        int filesSize = users.stream().mapToInt(u -> Objects.requireNonNull(u.getDirectory().listFiles()).length).sum();

        INDArray inputs = Nd4j.create(filesSize, IMG_TOTAL_SIZE);
        final INDArray labels = Nd4j.create(1, filesSize);
        int index = 0;
        for (User user : users) {
            int userIndex = users.indexOf(user);

            for (File image : Objects.requireNonNull(user.getDirectory().listFiles())) {
                INDArray imgArray = NATIVE_IMAGE_LOADER.asMatrix(image).reshape(IMG_TOTAL_SIZE).div(255);
                inputs.putRow(index, imgArray);
                labels.put(0, index, userIndex);
                imgArray.close();
                index++;
            }
        }
        final INDArray normalizedInputs = inputs.reshape(filesSize * (IMG_TOTAL_SIZE));
        final INDArray normalizedLabels = labels.reshape(filesSize);

        train(normalizedInputs, normalizedLabels, 100, 0.1);

        save();
    }

    @Override
    public void addUser(User user) {

    }

    @Override
    public void invalidate() {
        nativeInterface.deleteModel(model);
        this.model = null;
    }

    public void train(INDArray inputs, INDArray labels, int epoch, double alpha) {
        nativeInterface.trainModel(model, inputs.toDoubleVector(), labels.toDoubleVector(), (int) labels.length(), epoch, alpha);
    }

    public void save() {
        nativeInterface.saveModel(model, new File("models/mlp/neuralNetwork.model").getAbsolutePath());
    }
}
