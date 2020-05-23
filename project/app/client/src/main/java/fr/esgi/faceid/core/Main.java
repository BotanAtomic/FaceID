package fr.esgi.faceid.core;

import javafx.application.Application;
import javafx.fxml.FXMLLoader;
import javafx.scene.Parent;
import javafx.scene.Scene;
import javafx.stage.Stage;

import java.io.File;

/**
 * Created by Botan on 5/22/2020. 2:39 PM
 **/
public class Main extends Application {

    @Override
    public void start(Stage primaryStage) throws Exception {
        System.load(new File("libs/opencv_java430.dll").getAbsolutePath());
        Parent root = FXMLLoader.load(getClass().getResource("/root.fxml"));
        primaryStage.setScene(new Scene(root));
        primaryStage.show();
    }

}
