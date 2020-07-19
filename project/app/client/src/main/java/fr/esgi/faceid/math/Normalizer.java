package fr.esgi.faceid.math;

/**
 * Created by Botan on 5/30/2020. 10:03 PM
 **/
public class Normalizer {

    public static double softmax(double input, double[] neuronValues) {
        double totalLayerInput = 0;
        double max = 0;

        for (double v : neuronValues) {
            totalLayerInput += Math.exp(v - max);
        }

        return Math.exp(input - max) / totalLayerInput;
    }

}
