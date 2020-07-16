package fr.esgi.faceid.ai;

import fr.esgi.faceid.entity.User;
import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
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


/**
 * Created by Botan on 5/28/2020. 10:43 PM
 **/
public class DL4JCNNNetwork implements NeuralNetwork {

    private final static int IMG_SIZE = 32;

    private final static int IMG_CHANNEL = 3;

    public final static int IMG_TOTAL_SIZE = IMG_SIZE * IMG_SIZE * IMG_CHANNEL;

    public final NativeImageLoader NATIVE_IMAGE_LOADER = new NativeImageLoader(IMG_SIZE, IMG_SIZE, IMG_CHANNEL);

    private final List<User> users;

    private MultiLayerNetwork multiLayerNetwork;

    boolean trained = false;

    public DL4JCNNNetwork(List<User> users) {
        this.users = users;

        try {
            this.multiLayerNetwork = ModelSerializer.restoreMultiLayerNetwork(new File("models/dl4j/model_cnn.dl4j"));
            if(((OutputLayer)multiLayerNetwork.getOutputLayer()).getNOut() != users.size()) {
                throw new Exception("invalid configuration");
            }
            System.out.println("Restore DL4J model");
            trained = true;
        } catch (Exception e) {
            buildNetwork();
        }

    }

    private void buildNetwork() {
        if (users.size() > 1) {
            ConvolutionLayer layer0 = new ConvolutionLayer.Builder(5, 5)
                    .nIn(3)
                    .nOut(16)
                    .stride(1, 1)
                    .padding(2, 2)
                    .weightInit(WeightInit.XAVIER)
                    .activation(Activation.RELU)
                    .build();

            SubsamplingLayer layer1 = new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                    .kernelSize(2, 2)
                    .stride(2, 2)
                    .build();

            ConvolutionLayer layer2 = new ConvolutionLayer.Builder(5, 5)
                    .nOut(20)
                    .stride(1, 1)
                    .padding(2, 2)
                    .weightInit(WeightInit.XAVIER)
                    .activation(Activation.RELU)
                    .build();

            SubsamplingLayer layer3 = new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                    .kernelSize(2, 2)
                    .stride(2, 2)
                    .build();

            ConvolutionLayer layer4 = new ConvolutionLayer.Builder(5, 5)
                    .nOut(20)
                    .stride(1, 1)
                    .padding(2, 2)
                    .weightInit(WeightInit.XAVIER)
                    .activation(Activation.RELU)
                    .build();

            SubsamplingLayer layer5 = new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                    .kernelSize(2, 2)
                    .stride(2, 2)
                    .build();

            OutputLayer layer6 = new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                    .activation(Activation.SOFTMAX)
                    .weightInit(WeightInit.XAVIER)
                    .nOut(users.size())
                    .build();

            this.multiLayerNetwork = new MultiLayerNetwork(new NeuralNetConfiguration.Builder()
                    .weightInit(WeightInit.XAVIER)
                    .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                    .l2(0.0004)
                    .updater(new Adam())
                    .list()
                    .layer(0, layer0)
                    .layer(1, layer1)
                    .layer(2, layer2)
                    .layer(3, layer3)
                    .layer(4, layer4)
                    .layer(5, layer5)
                    .layer(6, layer6)
                    .setInputType(InputType.convolutional(IMG_SIZE, IMG_SIZE, IMG_CHANNEL))
                    .build()
            );
            trained = false;
        }
    }

    public User predict(Mat input) throws Exception {
        if (multiLayerNetwork == null || !trained) return null;

        int[] result = multiLayerNetwork.predict(NATIVE_IMAGE_LOADER.asMatrix(input).div(255));
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
        System.out.println("Train DL4J CNN");

        if (users.size() < 2) return;

        buildNetwork();
        multiLayerNetwork.init();

        int filesSize = users.stream().mapToInt(u -> Objects.requireNonNull(u.getDirectory().listFiles()).length).sum();

        INDArray inputs = Nd4j.create(filesSize, IMG_CHANNEL, IMG_SIZE, IMG_SIZE);
        final INDArray labels = Nd4j.create(filesSize, users.size());
        int index = 0;
        for (User user : users) {
            int userIndex = users.indexOf(user);

            for (File image : Objects.requireNonNull(user.getDirectory().listFiles())) {
                INDArray imgArray = NATIVE_IMAGE_LOADER.asMatrix(image).div(255);
                inputs.putRow(index, imgArray);
                labels.putRow(index, generateLabels(userIndex));
                index++;
            }
        }

        for (int i = 0; i < 300; i++)
            multiLayerNetwork.fit(inputs, labels);

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
}
