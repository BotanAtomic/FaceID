package fr.esgi.faceid.entity;

import com.sun.jna.Pointer;
import javafx.scene.image.Image;

import java.io.File;
import java.util.Random;

/**
 * Created by Botan on 5/28/2020. 10:42 PM
 **/

public class User {

    private String name;

    private File directory;

    private Pointer neuralNetwork;

    public User(File directory) {
        this.name = directory.getName();
        this.directory = directory;
    }

    public String getName() {
        return this.name;
    }

    public void setName(String name) {
        this.name = name;
    }

    public File getDirectory() {
        return this.directory;
    }

    public void setDirectory(File directory) {
        this.directory = directory;
    }

    public Pointer getNeuralNetwork() {
        return this.neuralNetwork;
    }

    public void setNeuralNetwork(Pointer neuralNetwork) {
        this.neuralNetwork = neuralNetwork;
    }

    public int countImages() {
        try {
            return directory.listFiles(file -> !file.getName().equals("model")).length;
        } catch (Exception e) {
            return 0;
        }
    }

    public Image randomImage() {
        File[] files = directory.listFiles();
        if (files != null && files.length > 0)
            return new Image(files[new Random().nextInt(files.length - 1)].toURI().toString());
        else
            return new Image(User.class.getResourceAsStream("/robot.png"));
    }
}
