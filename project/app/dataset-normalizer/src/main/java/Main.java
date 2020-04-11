import org.opencv.core.Mat;
import org.opencv.core.MatOfRect;
import org.opencv.core.Rect;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;
import org.opencv.objdetect.Objdetect;

import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;
import java.util.concurrent.Executor;
import java.util.concurrent.Executors;

/**
 * Created by Botan on 4/11/2020. 4:12 PM
 **/
public class Main {
    private final static Size faceSize = new Size(96, 96);

    private static boolean writeFace(CascadeClassifier faceCascade, Mat matrix, File file) {
        MatOfRect faces = new MatOfRect();
        faceCascade.detectMultiScale(matrix, faces, 1.1, 2, Objdetect.CASCADE_SCALE_IMAGE,
                new Size(200, 200), new Size());

        Rect[] facesArray = faces.toArray();

        if (facesArray.length > 0) {
            Rect firstFaceRect = facesArray[0];
            Rect faceRect = new Rect(firstFaceRect.x, firstFaceRect.y, firstFaceRect.width, firstFaceRect.height);
            Mat face = new Mat(matrix, faceRect);
            Imgproc.resize(face, face, faceSize);


            File newFile = new File(file.getAbsolutePath().replace("raw", "train"));
            if (newFile.exists())
                newFile.delete();
            try {
                Imgcodecs.imwrite(newFile.getAbsolutePath(), face);
                face.release();
                return true;
            } catch (Exception e) {

            }
        }
        return false;
    }

    private static void convertLabel(File source) {
        List<CascadeClassifier> classifierList = new ArrayList<>() {{
            add(new CascadeClassifier("models/face-detector/haarcascade_frontalface_alt.xml"));
            add(new CascadeClassifier("models/face-detector/haarcascade_frontalcatface_extended.xml"));
            add(new CascadeClassifier("models/face-detector/haarcascade_frontalface_alt.xml"));
            add(new CascadeClassifier("models/face-detector/haarcascade_frontalface_alt2.xml"));
            add(new CascadeClassifier("models/face-detector/haarcascade_frontalface_alt_tree.xml"));
            add(new CascadeClassifier("models/face-detector/haarcascade_frontalface_default.xml"));
        }};

        new File(source.getAbsolutePath().replace("raw", "train")).mkdirs();

        int faceFind = 0;
        int total = Objects.requireNonNull(source.listFiles()).length;
        System.out.println(String.format("Find %d images for label %s", total, source.getName()));
        for (File file : Objects.requireNonNull(source.listFiles())) {
            Mat matrix = Imgcodecs.imread(file.getAbsolutePath());
            if (!matrix.empty()) {
                for (CascadeClassifier cascadeClassifier : classifierList) {
                    if (writeFace(cascadeClassifier, matrix, file)) {
                        faceFind++;
                        break;
                    }
                }
                matrix.release();
                ;
            }
        }

        System.out.println(String.format("Extract %d faces of %d images for label %s", faceFind, total, source.getName()));
    }

    public static void main(String[] args) {
        Executor executor = Executors.newCachedThreadPool();
        System.load(new File("libs/opencv_java420.dll").getAbsolutePath());

        String directoryPath = "C:\\Users\\botan\\Work\\FaceID\\project\\dataset";
        System.out.println("Scanning directory <" + directoryPath + "> ...");
        File directory = new File(directoryPath);
        File trainDirectory = new File(directory.getAbsolutePath() + "\\train");
        File rawDirectory = new File(directory.getAbsolutePath() + "\\raw");


        if (!rawDirectory.exists()) {
            System.out.println("Cannot find raw directory -> " + directory.getAbsolutePath() + "\raw");
            System.exit(-1);
        }

        trainDirectory.mkdirs();

        List<File> toConvert = new ArrayList<>();

        for (File file : Objects.requireNonNull(rawDirectory.listFiles())) {
            if (file.isDirectory())
                toConvert.add(file);
        }

        System.out.println(String.format("Find %d labels %s, starting normalization", toConvert.size(), Arrays.toString(toConvert.stream().map(File::getName).toArray())));

        toConvert.forEach(f -> executor.execute(() -> convertLabel(f)));

    }

}
