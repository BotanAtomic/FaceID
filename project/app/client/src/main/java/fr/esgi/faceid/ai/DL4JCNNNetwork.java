package fr.esgi.faceid.ai;

import fr.esgi.faceid.entity.User;
import javafx.util.Pair;
import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.core.storage.StatsStorage;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.ui.model.stats.StatsListener;
import org.deeplearning4j.ui.model.storage.InMemoryStatsStorage;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.opencv.core.Mat;

import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;

import static fr.esgi.faceid.ai.NeuralNetworkManager.uiServer;


/**
 * Created by Botan on 5/28/2020. 10:43 PM
 **/
public class DL4JCNNNetwork implements NeuralNetwork {

    private final static int IMG_SIZE = 28;

    private final static int IMG_CHANNEL = 1;

    public final static int IMG_TOTAL_SIZE = IMG_SIZE * IMG_SIZE * IMG_CHANNEL;

    public final NativeImageLoader NATIVE_IMAGE_LOADER = new NativeImageLoader(IMG_SIZE, IMG_SIZE, IMG_CHANNEL);

    private final List<User> users;
    boolean trained = false;
    private MultiLayerNetwork multiLayerNetwork;

    public DL4JCNNNetwork(List<User> users) {
        this.users = users;

        try {
            this.multiLayerNetwork = ModelSerializer.restoreMultiLayerNetwork(new File("models/dl4j/model_cnn.dl4j"));
            System.out.println("Restore DL4J model");
            trained = true;
        } catch (Exception e) {
            e.printStackTrace();
            buildNetwork();
        }

    }

    private void buildNetwork() {
        if (users.size() > 1) {
            ConvolutionLayer layer0 = new ConvolutionLayer.Builder(5, 5)
                    .nIn(3)
                    .nOut(16)
                    .activation(Activation.RELU)
                    .build();

            SubsamplingLayer layer1 = new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                    .kernelSize(2, 2)
                    .build();

            ConvolutionLayer layer2 = new ConvolutionLayer.Builder(5, 5)
                    .nOut(20)
                    .activation(Activation.RELU)
                    .build();

            SubsamplingLayer layer3 = new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                    .kernelSize(2, 2)
                    .build();

            ConvolutionLayer layer4 = new ConvolutionLayer.Builder(5, 5)
                    .nOut(20)
                    .activation(Activation.RELU)
                    .build();

            SubsamplingLayer layer5 = new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                    .kernelSize(2, 2)
                    .build();

            OutputLayer layer6 = new OutputLayer.Builder(LossFunctions.LossFunction.RECONSTRUCTION_CROSSENTROPY)
                    .activation(Activation.SOFTMAX)
                    .nOut(users.size())
                    .build();

            this.multiLayerNetwork = new MultiLayerNetwork(new NeuralNetConfiguration.Builder()
                    .weightInit(WeightInit.XAVIER)
                    .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                    .l2(0.0004)
                    .activation(Activation.RELU)
                    .updater(new Adam())
                    .list()
                    .layer(0, convInit("cnn1", IMG_CHANNEL, 32, new int[]{5, 5}, new int[]{1, 1}, new int[]{0, 0}, 0))
                    .layer(1, maxPool("maxpool1", new int[]{2, 2}))
                    .layer(2, conv3x3("cnn2", 64, 0))
                    .layer(3, conv3x3("cnn3", 64, 1))
                    .layer(4, maxPool("maxpool2", new int[]{2, 2}))
                    .layer(5, new DenseLayer.Builder().activation(Activation.RELU)
                            .nOut(512).dropOut(0.5).build())
                    .layer(6, new OutputLayer.Builder(LossFunctions.LossFunction.RECONSTRUCTION_CROSSENTROPY)
                            .nOut(users.size())
                            .activation(Activation.SOFTMAX)
                            .build())
                    .setInputType(InputType.convolutional(IMG_SIZE, IMG_SIZE, IMG_CHANNEL))
                    .build()
            );
            StatsStorage statsStorage = new InMemoryStatsStorage();
            uiServer.attach(statsStorage);
            multiLayerNetwork.setListeners(new StatsListener(statsStorage));
            trained = false;
        }
    }

    public Pair<User, Integer> predict(Mat input) throws Exception {
        if (multiLayerNetwork == null || !trained) return null;

        INDArray result = multiLayerNetwork.output(NATIVE_IMAGE_LOADER.asMatrix(input).div(255));
        int index = (int) result.argMax(1).toDoubleVector()[0];
        double probability = result.toDoubleVector()[index] * 100;

        if(probability < 80) return null;

        return new Pair<>(users.get(index), (int) probability);
    }

    private INDArray generateLabels(int pos) {
        double[] label = new double[users.size()];
        Arrays.fill(label, 0);
        label[pos] = 1;
        return Nd4j.create(label);
    }

    @Override
    public void train() throws Exception {
        System.out.println("Train DL4J CNN");

        if (users.size() < 2) return;

        buildNetwork();
        multiLayerNetwork.init();

        int filesSize = users.stream().mapToInt(u -> Objects.requireNonNull(u.getDirectory().listFiles()).length).sum();

        System.out.println("Loading dataset...");

        INDArray inputs = Nd4j.create(filesSize, IMG_CHANNEL, IMG_SIZE, IMG_SIZE);
        final INDArray labels = Nd4j.create(filesSize, users.size());
        int index = 0;
        for (User user : users) {
            int userIndex = users.indexOf(user);

            int i = 0;
            for (File image : Objects.requireNonNull(user.getDirectory().listFiles())) {
                if (i > 200) break;
                INDArray imgArray = NATIVE_IMAGE_LOADER.asMatrix(image).div(255);
                inputs.putRow(index, imgArray);
                labels.putRow(index, generateLabels(userIndex));
                index++;
                i++;
            }
        }

        System.out.println("Dataset loaded");

        for (int i = 0; i < 150; i++) {
            multiLayerNetwork.fit(inputs, labels);
            System.out.println("Epohs " + i + ": done.");
        }
        save();
    }

    @Override
    public void addUser(User u) {

    }

    @Override
    public void invalidate() {
        this.multiLayerNetwork = null;
    }

    public void save() throws IOException {
        multiLayerNetwork.save(new File("models/dl4j/model_cnn.dl4j"));
        trained = true;
    }

    private ConvolutionLayer convInit(String name, int in, int out, int[] kernel, int[] stride, int[] pad, double bias) {
        return new ConvolutionLayer.Builder(kernel, stride, pad).name(name).nIn(in).nOut(out).biasInit(bias).build();
    }

    private ConvolutionLayer conv3x3(String name, int out, double bias) {
        return new ConvolutionLayer.Builder(new int[]{3, 3}, new int[]{1, 1}, new int[]{1, 1}).name(name).nOut(out).biasInit(bias).build();
    }


    private SubsamplingLayer maxPool(String name, int[] kernel) {
        return new SubsamplingLayer.Builder(kernel, new int[]{2, 2}).name(name).build();
    }
}
