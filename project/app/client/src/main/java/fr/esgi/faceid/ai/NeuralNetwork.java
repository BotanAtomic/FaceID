package fr.esgi.faceid.ai;

import fr.esgi.faceid.entity.User;
import org.opencv.core.Mat;

public interface NeuralNetwork {

    User predict(Mat input) throws Exception;

    void train() throws Exception;

    void addUser(User user);

    void invalidate();
}
