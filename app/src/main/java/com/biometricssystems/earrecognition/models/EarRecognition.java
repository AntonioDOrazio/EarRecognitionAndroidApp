package com.biometricssystems.earrecognition.models;

import android.content.Context;
import android.content.SharedPreferences;
import android.graphics.Bitmap;
import android.widget.Toast;

import com.biometricssystems.earrecognition.R;
import com.biometricssystems.earrecognition.ml.Model;

import org.apache.commons.math3.stat.correlation.PearsonsCorrelation;
import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfDouble;
import org.opencv.imgproc.Imgproc;
import org.tensorflow.lite.DataType;
import org.tensorflow.lite.support.common.ops.NormalizeOp;
import org.tensorflow.lite.support.image.ImageProcessor;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.image.ops.ResizeOp;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Locale;

public class EarRecognition {

    Model feat_extractor;
    double THRESHOLD = 0.98;
    double similarityAchieved = 0;
    boolean verificationSuccess;
    ArrayList<double[]> templates;

    public EarRecognition(Context context, SharedPreferences sharedPrefs){
        if(sharedPrefs!=null)
            templates = retrieveTemplates(sharedPrefs);
        if(feat_extractor == null) {
            try {
                feat_extractor = Model.newInstance(context);
            } catch (IOException e) {
                // TODO Handle the exception
            }
            Toast.makeText(context, "EarRecognition: feature extractor initialized", Toast.LENGTH_LONG).show();
        }
    }

    public double getSimilarityAchieved() {
        return similarityAchieved;
    }

    public boolean isVerificationSuccess() {
        return verificationSuccess;
    }

    public void extractAndSaveFeatures(Bitmap segmentedEar, int index, SharedPreferences sharedPrefs){
        Mat frame = new Mat();
        Utils.bitmapToMat(segmentedEar, frame);

        Imgproc.cvtColor(frame, frame, Imgproc.COLOR_RGBA2RGB);
        MatOfDouble mean = new MatOfDouble();
        MatOfDouble std = new MatOfDouble();
        Core.meanStdDev(frame, mean, std);
        float[] m = new float[]{
                (float)mean.get(0,0)[0],
                (float)mean.get(1,0)[0],
                (float)mean.get(2,0)[0]};
        float[] stdev = new float[]{
                (float)std.get(0,0)[0],
                (float)std.get(1,0)[0],
                (float)std.get(2,0)[0]};

        ImageProcessor imageProcessor = new ImageProcessor.Builder()
                .add(new ResizeOp(224, 224, ResizeOp.ResizeMethod.BILINEAR))
                .add(new NormalizeOp(m, stdev))
                .build();

        TensorImage tensorImage = new TensorImage(DataType.FLOAT32);

        try {
            tensorImage.load(segmentedEar);

            tensorImage = imageProcessor.process(tensorImage);
        }catch (Exception e){
            System.err.println(e.getMessage());
        }
        // Runs model inference and gets result.
        float[] output = feat_extractor.process(tensorImage.getTensorBuffer())
                .getOutputFeature0AsTensorBuffer()
                .getFloatArray();

        String[] features = new String[1280];

        for(int i=0; i<1280; i++){
            features[i] = String.valueOf(output[i]);
        }

        String joined = String.join(",", features);

        SharedPreferences.Editor editor = sharedPrefs.edit();
        String str = String.format(Locale.getDefault(), "t%d", index);
        editor.putString(str, joined);
        //System.out.println("EarRecognition: saving " + str + joined);
        editor.apply();
    }

    public boolean performVerification(Bitmap croppedEar){
        verificationSuccess = false;
        if(templates.size()<3){
            return false;
        }
        Mat frame = new Mat();
        Utils.bitmapToMat(croppedEar, frame);

        Imgproc.cvtColor(frame, frame, Imgproc.COLOR_RGBA2RGB);
        MatOfDouble mean = new MatOfDouble();
        MatOfDouble std = new MatOfDouble();
        Core.meanStdDev(frame, mean, std);
        float[] m = new float[]{
                (float)mean.get(0,0)[0],
                (float)mean.get(1,0)[0],
                (float)mean.get(2,0)[0]};
        float[] stdev = new float[]{
                (float)std.get(0,0)[0],
                (float)std.get(1,0)[0],
                (float)std.get(2,0)[0]};

        ImageProcessor imageProcessor = new ImageProcessor.Builder()
                .add(new ResizeOp(224, 224, ResizeOp.ResizeMethod.BILINEAR))
                .add(new NormalizeOp(m, stdev))
                .build();

        TensorImage tensorImage = new TensorImage(DataType.FLOAT32);

        try {
            tensorImage.load(croppedEar);

            tensorImage = imageProcessor.process(tensorImage);
        }catch (Exception e){
            System.err.println(e.getMessage());
        }
        // Runs model inference and gets result.
        float[] output = feat_extractor.process(tensorImage.getTensorBuffer())
                .getOutputFeature0AsTensorBuffer()
                .getFloatArray();

        double[] probe = new double[1280];
        for (int i = 0; i < 1280; i++) {
            probe[i] = output[i];
        }

        similarityAchieved = calculateSimilarity(probe, templates);
        verificationSuccess = similarityAchieved > THRESHOLD;

        return  true;
    }

    private double calculateSimilarity(double[] probe, ArrayList<double[]> templates){
        double maxSimilarity = 0;
        PearsonsCorrelation corr = new PearsonsCorrelation();
        for(double[] template:templates){
            double similarity = corr.correlation(probe, template);
            if(similarity > maxSimilarity)
                maxSimilarity = similarity;
        }
        return maxSimilarity;
    }

    private ArrayList<double[]> retrieveTemplates(SharedPreferences sharedPrefs){
        ArrayList<String> featureStrings = new ArrayList<>();
        featureStrings.add(sharedPrefs.getString("t0", null));
        featureStrings.add(sharedPrefs.getString("t1", null));
        featureStrings.add(sharedPrefs.getString("t2", null));

        ArrayList<double[]> featuresRetrieved = new ArrayList<>();
        for(String feature : featureStrings){
            if(feature!=null){
                String[] strings = feature.split(",");
                double[] values = new double[1280];
                for(int i=0; i<1280; i++)
                    values[i] = Double.parseDouble(strings[i]);
                featuresRetrieved.add(values);
            }
        }
        return featuresRetrieved;
    }
}
