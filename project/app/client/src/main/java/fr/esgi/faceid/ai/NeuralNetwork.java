package fr.esgi.faceid.ai;

import com.sun.jna.Pointer;
import fr.esgi.faceid.api.ILinearModel;
import fr.esgi.faceid.configuration.Configuration;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.io.File;

/**
 * Created by Botan on 5/28/2020. 10:43 PM
 **/
public class NeuralNetwork {

    private final int modelSize = Configuration.IMG_TOTAL_SIZE;

    private final ILinearModel nativeInterface;

    private Pointer model;

    public NeuralNetwork(int size, ILinearModel nativeInterface) {
        this.model = nativeInterface.createModel(size);
        this.nativeInterface = nativeInterface;
    }

    public NeuralNetwork(Pointer model, ILinearModel nativeInterface) {
        this.model = model;
        this.nativeInterface = nativeInterface;
    }

    public static NeuralNetwork from(File modelFile, ILinearModel nativeInterface) {
        return new NeuralNetwork(nativeInterface.loadModel(modelFile.getAbsolutePath()), nativeInterface);
    }

    public double predict(double[] data) {
        return nativeInterface.predictRegressionModel(model, data, data.length);
    }

    public void train(INDArray inputs, INDArray labels, int length, int dataSize, int epoch, double alpha) {
        nativeInterface.trainModel(model, inputs.toDoubleVector(), labels.toDoubleVector(), length, dataSize, epoch, alpha);
    }

    public void save(File file) {
        nativeInterface.saveModel(model, modelSize, file.getAbsolutePath());
    }
}
