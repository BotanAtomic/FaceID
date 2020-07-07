package fr.esgi.faceid.ai;

import fr.esgi.faceid.entity.User;
import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.opencv.core.Mat;

import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;


/**
 * Created by Botan on 5/28/2020. 10:43 PM
 **/
public class DL4JNetwork implements NeuralNetwork {

    private final static int IMG_SIZE = 28;

    private final static int IMG_CHANNEL = 1;

    public final static int IMG_TOTAL_SIZE = IMG_SIZE * IMG_SIZE * IMG_CHANNEL;

    public final NativeImageLoader NATIVE_IMAGE_LOADER = new NativeImageLoader(IMG_SIZE, IMG_SIZE, IMG_CHANNEL);

    private final List<User> users;

    private MultiLayerNetwork multiLayerNetwork;

    public DL4JNetwork(List<User> users) {
        this.users = users;

        try {
            this.multiLayerNetwork = ModelSerializer.restoreMultiLayerNetwork(new File("models/dl4j/model.dl4j"));
            System.out.println("Restore DL4J model");
        } catch (IOException e) {
            this.multiLayerNetwork = new MultiLayerNetwork(new NeuralNetConfiguration.Builder()
                    .weightInit(WeightInit.XAVIER)
                    .updater(new Adam(0.01))
                    .list()
                    .layer(0, new DenseLayer.Builder().nIn(IMG_TOTAL_SIZE).nOut(128).activation(Activation.RELU).build())
                    .layer(1, new DenseLayer.Builder().nIn(128).nOut(128).activation(Activation.SIGMOID).build())
                    .layer(2, new OutputLayer.Builder().nIn(128).nOut(users.size()).activation(Activation.SIGMOID).build())
                    .build());
        }

        multiLayerNetwork.init();
        multiLayerNetwork.setListeners(new ScoreIterationListener(10));
    }

    public User predict(Mat input) throws Exception {
        int[] result = multiLayerNetwork.predict(NATIVE_IMAGE_LOADER.asMatrix(input).div(255).reshape(new int[]{1, IMG_TOTAL_SIZE}));
        return users.get(result[0]);
    }

    private INDArray generateLabels(int pos) {
        double[] label = new double[users.size()];
        Arrays.fill(label, 0);
        label[pos] = 1;
        return Nd4j.create(label);
    }

    @Override
    public void train() throws Exception {
        System.out.println("Train DL4J");

        int filesSize = users.stream().mapToInt(u -> Objects.requireNonNull(u.getDirectory().listFiles()).length).sum();

        INDArray inputs = Nd4j.create(filesSize, IMG_TOTAL_SIZE);
        final INDArray labels = Nd4j.create(filesSize, users.size());
        int index = 0;
        for (User user : users) {
            int userIndex = users.indexOf(user);

            for (File image : Objects.requireNonNull(user.getDirectory().listFiles())) {
                INDArray imgArray = NATIVE_IMAGE_LOADER.asMatrix(image).reshape(IMG_TOTAL_SIZE).div(255);
                inputs.putRow(index, imgArray);
                labels.putRow(index, generateLabels(userIndex));
                index++;
            }
        }

        for (int i = 0; i < 100; i++)
            multiLayerNetwork.fit(inputs, labels);

        save();
    }

    @Override
    public void addUser(User u) {

    }


    public void save() throws IOException {
        multiLayerNetwork.save(new File("models/dl4j/model.dl4j"));
    }
}
