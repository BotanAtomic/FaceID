package fr.esgi.faceid.math;

import java.util.Arrays;

/**
 * Created by Botan on 5/30/2020. 10:03 PM
 **/
public class Normalizer {

    public static double softmax(double input, double[] neuronValues) {
        double total = Arrays.stream(neuronValues).map(Math::exp).sum();
        return Math.exp(input) / total;
    }

}
