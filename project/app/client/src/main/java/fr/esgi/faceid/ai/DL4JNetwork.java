package fr.esgi.faceid.ai;

import fr.esgi.faceid.entity.User;
import javafx.util.Pair;
import org.datavec.api.io.filters.BalancedPathFilter;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.datavec.image.transform.ScaleImageTransform;
import org.deeplearning4j.core.storage.StatsStorage;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.datasets.iterator.DataSetIteratorSplitter;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.Convolution3D;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.DropoutLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.ui.model.stats.StatsListener;
import org.deeplearning4j.ui.model.storage.InMemoryStatsStorage;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerMinMaxScaler;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.learning.regularization.L2Regularization;
import org.opencv.core.Mat;

import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;
import java.util.Random;

import static fr.esgi.faceid.ai.NeuralNetworkManager.uiServer;


/**
 * Created by Botan on 5/28/2020. 10:43 PM
 **/
public class DL4JNetwork implements NeuralNetwork {

    private final static int IMG_SIZE = 96;

    private final static int IMG_CHANNEL = 3;

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
            buildNetwork();
        }
        multiLayerNetwork.init();
    }

    private void buildNetwork() {
        if (users.size() > 1) {
            this.multiLayerNetwork = new MultiLayerNetwork(new NeuralNetConfiguration.Builder()
                    .weightInit(WeightInit.XAVIER)
                    .updater(new Adam())
                    .list()
                    .layer(new DenseLayer.Builder().nIn(IMG_TOTAL_SIZE).nOut(512).activation(Activation.RELU).build())
                    .layer(new DenseLayer.Builder().nIn(512).nOut(256).activation(Activation.RELU).build())
                    .layer(new DenseLayer.Builder().nIn(256).nOut(128).activation(Activation.RELU).build())
                    .layer(new DenseLayer.Builder().nIn(128).nOut(64).activation(Activation.RELU).build())
                    .layer(new OutputLayer.Builder().nIn(64).nOut(users.size()).activation(Activation.SOFTMAX).build())
                    .setInputType(InputType.convolutional(IMG_SIZE, IMG_SIZE, IMG_CHANNEL))
                    .build());
            StatsStorage statsStorage = new InMemoryStatsStorage();
            uiServer.attach(statsStorage);
            multiLayerNetwork.setListeners(new StatsListener(statsStorage));
        }
    }

    public Pair<User, Integer> predict(Mat input) throws Exception {
        if (multiLayerNetwork == null) return null;

        INDArray result = multiLayerNetwork.output(NATIVE_IMAGE_LOADER.asMatrix(input).div(255).reshape(new int[]{1, IMG_TOTAL_SIZE}));
        int index = (int) result.argMax(1).toDoubleVector()[0];
        double probability = result.toDoubleVector()[index] * 100;

        if (probability < 80) return null;

        return new Pair<>(users.get(index), (int) probability);
    }

    private INDArray generateLabels(int pos) {
        double[] label = new double[users.size()];
        Arrays.fill(label, 0);
        label[pos] = 1;
        return Nd4j.create(label);
    }

    /**
     *  Accuracy:        0.9849
     *  Precision:       0.9853
     *  Recall:          0.9849
     *  F1 Score:        0.9849
     * @throws Exception
     */

    @Override
    public void train() throws Exception {
        System.out.println("Train DL4J");

        if (users.size() < 2) return;

        buildNetwork();
        multiLayerNetwork.init();

        File parentDir = new File("data/");

        var labelMaker = new ParentPathLabelGenerator();
        var recordReader = new ImageRecordReader(IMG_SIZE, IMG_SIZE, IMG_CHANNEL, labelMaker);

        BalancedPathFilter pathFilter = new BalancedPathFilter(new Random(123), NativeImageLoader.ALLOWED_FORMATS, labelMaker);
        InputSplit[] inputSplits = new FileSplit(parentDir).sample(pathFilter, 0.80, 0.20);
        recordReader.initialize(inputSplits[0]);

        DataSetIterator dataIter = new RecordReaderDataSetIterator(recordReader, 32, 1, users.size());
        var preProcessor = new ImagePreProcessingScaler();
        preProcessor.fit(dataIter);
        dataIter.setPreProcessor(preProcessor);

        multiLayerNetwork.fit(dataIter, 2);

        recordReader.initialize(inputSplits[1]);
        dataIter = new RecordReaderDataSetIterator(recordReader, 32, 1, users.size());
        preProcessor.fit(dataIter);
        dataIter.setPreProcessor(preProcessor);

        var evaluation = multiLayerNetwork.evaluate(dataIter);
        System.out.println(evaluation.stats(true, true));
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
        multiLayerNetwork.save(new File("models/dl4j/model.dl4j"));
    }
}
