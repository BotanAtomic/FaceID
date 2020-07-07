package fr.esgi.faceid.api;

import com.sun.jna.Library;
import com.sun.jna.Pointer;

/**
 * Created by Botan on 5/22/2020. 4:49 PM
 **/
public interface IMultiLayerNetwork extends Library {

    Pointer createModel(int size);

    Pointer addLayer(Pointer model, int neuron, String layer);

    Pointer predict(Pointer model, double[] inputs);

    void trainModel(Pointer model, double[] inputs, double[] labels, int size, int epoch, double alpha);

    Pointer saveModel(Pointer model, String path);

    Pointer loadModel(String path);

    void deleteModel(Pointer model);

}
