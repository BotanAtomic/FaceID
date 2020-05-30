package fr.esgi.faceid.api;

import com.sun.jna.Library;
import com.sun.jna.Pointer;

/**
 * Created by Botan on 5/22/2020. 4:49 PM
 **/
public interface ILinearModel extends Library {

    Pointer createModel(int size);

    double predictRegressionModel(Pointer model, double[] inputs, int size);

    double trainModel(Pointer model, double[] inputs, double[] labels, int size, int inputSize, int epoch, double alpha);

    Pointer saveModel(Pointer model, int size, String path);

    Pointer loadModel(String path);

    void deleteModel(Pointer model);

}
