import org.opencv.core.Mat;
import org.opencv.core.MatOfRect;
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
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Created by Botan on 4/11/2020. 4:12 PM
 **/
public class Main {
    private static Size faceSize = new Size(48, 48);
    private static Size minSize = new Size(400, 400);

    private static boolean writeFace(CascadeClassifier faceCascade, Mat matrix, File file) {
        MatOfRect faces = new MatOfRect();
        faceCascade.detectMultiScale(matrix, faces, 1.1, 1, Objdetect.CASCADE_SCALE_IMAGE,
                minSize, new Size());

        AtomicBoolean found = new AtomicBoolean(false);
        faces.toList().stream().min((o1, o2) -> o2.width - o1.width).ifPresent(rect -> {
            Mat face = new Mat(matrix, rect);
            Imgproc.resize(face, face, faceSize);

            File newFile = new File(file.getAbsolutePath().replace("raw", "train"));

            if (newFile.exists())
                newFile.delete();
            try {
                Imgcodecs.imwrite(newFile.getAbsolutePath(), face);
                face.release();
                found.set(true);
            } catch (Exception e) {
                e.printStackTrace();
            }
        });

        return found.get();
    }

    private static void convertLabel(File source) {
        List<CascadeClassifier> classifierList = new ArrayList<>() {{
            add(new CascadeClassifier("../../models/face-detector/haarcascade_frontalcatface_extended.xml"));
            add(new CascadeClassifier("../../models/face-detector/haarcascade_frontalface_default.xml"));
            add(new CascadeClassifier("../../models/face-detector/haarcascade_frontalface_alt.xml"));
            add(new CascadeClassifier("../../models/face-detector/haarcascade_frontalface_alt2.xml"));
            add(new CascadeClassifier("../../models/face-detector/haarcascade_frontalface_alt_tree.xml"));
        }};

        new File(source.getAbsolutePath().replace("raw", "train")).mkdirs();

        final AtomicInteger faceFind = new AtomicInteger();
        int total = Objects.requireNonNull(source.listFiles()).length;
        System.out.println(String.format("Find %d images for label %s", total, source.getName()));
        for (File file : Objects.requireNonNull(source.listFiles())) {
            Mat matrix = Imgcodecs.imread(file.getAbsolutePath());
            if (!matrix.empty()) {
                classifierList.stream().flatMap(classifier -> {
                    MatOfRect faces = new MatOfRect();
                    classifier.detectMultiScale(matrix, faces, 1.1, 1, Objdetect.CASCADE_SCALE_IMAGE,
                            minSize, new Size());
                    return Arrays.stream(faces.toArray());
                }).min((o1, o2) -> o2.width - o1.width).ifPresent(rect -> {
                    Mat face = new Mat(matrix, rect);
                    Imgproc.resize(face, face, faceSize);

                    File newFile = new File(file.getAbsolutePath().replace("raw", "train"));

                    if (newFile.exists())
                        newFile.delete();
                    try {
                        Imgcodecs.imwrite(newFile.getAbsolutePath(), face);
                        face.release();
                        faceFind.incrementAndGet();
                    } catch (Exception e) {
                        e.printStackTrace();
                    }
                });
                matrix.release();
            }
        }

        System.out.println(String.format("Extract %d faces of %d images for label %s", faceFind.get(), total, source.getName()));
    }

    public static void main(String[] args) {
        Executor executor = Executors.newCachedThreadPool();
        System.load(new File("libs/opencv_java420.dll").getAbsolutePath());

        String directoryPath = "C:\\Users\\botan\\Work\\FaceID\\project\\dataset";

        for (String arg : args) {
            String[] split = arg.split("=");
            if (split[0].toLowerCase().equals("--path")) {
                directoryPath = split[1];
            } else if (split[0].toLowerCase().equals("--size")) {
                faceSize = new Size(Integer.parseInt(split[1]), Integer.parseInt(split[1]));
            } else if (split[0].toLowerCase().equals("--min-size")) {
                minSize = new Size(Integer.parseInt(split[1]), Integer.parseInt(split[1]));
            }
        }

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
