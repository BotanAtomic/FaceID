package fr.esgi.faceid.core;

import javafx.application.Application;
import javafx.fxml.FXMLLoader;
import javafx.scene.Parent;
import javafx.scene.Scene;
import javafx.stage.Stage;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.factory.Nd4j;

import java.io.File;

/**
 * Created by Botan on 5/22/2020. 2:39 PM
 **/
public class Main extends Application {

    public static final String LIB_PATH = new File("..\\..\\lib\\linear-model\\cmake-build-release\\ML-framework.dll").getAbsolutePath();
    public static final String MLP_LIB_PATH = new File("..\\..\\lib\\multilayer-perceptron\\cmake-build-release-visual-studio\\ML-framework.dll").getAbsolutePath();

    @Override
    public void start(Stage primaryStage) throws Exception {
        System.out.println(LIB_PATH);
        Nd4j.setDefaultDataTypes(DataType.DOUBLE, DataType.DOUBLE);
        new File("data").mkdir();
        new File("models").mkdir();
        System.load(new File("libs/opencv_java430.dll").getAbsolutePath());
        Parent root = FXMLLoader.load(getClass().getResource("/root.fxml"));
        primaryStage.setScene(new Scene(root));
        primaryStage.show();
    }

}
