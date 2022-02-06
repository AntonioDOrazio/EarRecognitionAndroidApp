package com.biometricssystems.earrecognition;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;

import android.Manifest;
import android.content.Context;
import android.content.SharedPreferences;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.os.Bundle;
import android.os.Handler;
import android.util.Log;
import android.view.SurfaceView;
import android.view.View;
import android.widget.Button;
import android.widget.TextView;
import android.widget.Toast;

import com.biometricssystems.earrecognition.ml.Model;

import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.stat.correlation.PearsonsCorrelation;
import org.apache.commons.math3.util.MathArrays;
import org.apache.commons.math3.util.MathUtils;
import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.JavaCameraView;
import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
import org.opencv.core.MatOfDouble;
import org.opencv.core.MatOfFloat;
import org.opencv.core.MatOfInt;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfRect2d;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Rect2d;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.dnn.Net;
import org.opencv.imgproc.Imgproc;
import org.opencv.dnn.Dnn;
import org.opencv.utils.Converters;
import org.tensorflow.lite.DataType;
import org.tensorflow.lite.support.common.ops.NormalizeOp;
import org.tensorflow.lite.support.image.ImageProcessor;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.image.ops.ResizeOp;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.Charset;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Locale;

public class VerificationActivity extends AppCompatActivity implements CameraBridgeViewBase.CvCameraViewListener2 {

    CameraBridgeViewBase cameraBridgeViewBase;
    SharedPreferences sharedPrefs=null;

    public static Handler handler = new Handler();

    boolean startYolo = false;
    boolean firstTimeYolo = true;
    boolean verificationSuccess = false;
    Net tinyYolo;
    Model feat_extractor;
    Bitmap croppedEar = null;
    double THRESHOLD = 0.98;
    ArrayList<double[]> templates = null;

    Mat oldFrame; // Initialization is later in the code
    private double similarityAchieved;

    TextView verifTitle;

    private byte[] loadTextFromAssets(String assetsPath, Charset charset) throws IOException {
        InputStream is = getAssets().open(assetsPath);
        byte[] buffer = new byte[1024];
        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        for (int length = is.read(buffer); length != -1; length = is.read(buffer)) {
            baos.write(buffer, 0, length);
        }
        is.close();
        baos.close();
        return charset == null ? baos.toByteArray() : baos.toByteArray() ;
    }

    private boolean isNotMoving(Mat curFrame_) {

        Mat curFrame = curFrame_;

        // If not oldFrame, preprocess oldFrame and return not moving
        if (oldFrame == null) {
            oldFrame = new Mat();
            Imgproc.cvtColor(curFrame, oldFrame, Imgproc.COLOR_RGBA2GRAY);
            Imgproc.blur(oldFrame, oldFrame, new Size(21,21));
            return true;
        }

        Imgproc.cvtColor(curFrame, curFrame, Imgproc.COLOR_RGBA2GRAY);
        Imgproc.blur(curFrame, curFrame, new Size(21,21));

        Mat deltaframe = new Mat();
        Core.absdiff(oldFrame, curFrame, deltaframe);

        Mat thresholdFrame = new Mat();
        double threshold = Imgproc.threshold(deltaframe, thresholdFrame, 25, 255, Imgproc.THRESH_BINARY);

        Imgproc.dilate(thresholdFrame, thresholdFrame, new Mat());

        List<MatOfPoint> contours = new ArrayList<>();
        Mat hierarchy = new Mat();
        Imgproc.findContours(thresholdFrame, contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);

        double contourArea = 0;
        for (int i = 0; i < contours.size(); i++) {
            contourArea += Imgproc.contourArea(contours.get(i));
        }

        System.out.println("##########");
        System.out.println("Threshold: "    + threshold);
        System.out.println("Contour Area: " + contourArea);

        oldFrame = curFrame.clone();

        return(contourArea < 500); // old 100 poi 120

    }

    private void Yolo(View button) {

        if (!startYolo) {
            startYolo = true;
            if (firstTimeYolo) {
                MatOfByte yoloWeights = new MatOfByte();
                try {
                    yoloWeights.fromArray(loadTextFromAssets("ears.onnx", null));
                } catch (IOException e) {
                    e.printStackTrace();
                }

                tinyYolo = Dnn.readNetFromONNX(yoloWeights);
                firstTimeYolo = false;
                Toast.makeText(this, "yolo initialized", Toast.LENGTH_SHORT).show();

                initFeatureExtractor();
            }

        } else {
            startYolo = false;
        }
    }

    private void initFeatureExtractor() {
        if(feat_extractor == null) {
            try {
                feat_extractor = Model.newInstance(this);
            } catch (IOException e) {
                // TODO Handle the exception
            }
            Toast.makeText(this, "EarRecognition: feature extractor initialized", Toast.LENGTH_LONG).show();
        }
    }

    private static final int MY_CAMERA_REQUEST_CODE = 100;
    private static final int MY_READ_STORAGE_REQUEST_CODE = 101;
    private static final int MY_WRITE_STORAGE_REQUEST_CODE = 102;

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (requestCode == MY_CAMERA_REQUEST_CODE) {
            if (grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                Toast.makeText(this, "camera permission granted", Toast.LENGTH_LONG).show();


            } else {
                Toast.makeText(this, "camera permission denied", Toast.LENGTH_LONG).show();
            }
        }
    }


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_verification);

        if(!OpenCVLoader.initDebug()){
            Log.d("CVerror","OpenCV library Init failure");
        }else{
            // load your library and do initializing stuffs like System.loadLibrary();
        }

        cameraBridgeViewBase = (JavaCameraView) findViewById((R.id.CameraView));
        cameraBridgeViewBase.setVisibility(SurfaceView.VISIBLE);
        cameraBridgeViewBase.setCvCameraViewListener(this);
        cameraBridgeViewBase.setCameraIndex(1);
        sharedPrefs = getSharedPreferences(getString(R.string.app_name), Context.MODE_PRIVATE);

        if (checkSelfPermission(Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
            requestPermissions(new String[]{Manifest.permission.CAMERA}, MY_CAMERA_REQUEST_CODE);
        }
        if (checkSelfPermission(Manifest.permission.READ_EXTERNAL_STORAGE) != PackageManager.PERMISSION_GRANTED) {
            requestPermissions(new String[]{Manifest.permission.READ_EXTERNAL_STORAGE}, MY_READ_STORAGE_REQUEST_CODE);
        }
        if (checkSelfPermission(Manifest.permission.WRITE_EXTERNAL_STORAGE) != PackageManager.PERMISSION_GRANTED) {
            requestPermissions(new String[]{Manifest.permission.WRITE_EXTERNAL_STORAGE}, MY_WRITE_STORAGE_REQUEST_CODE);
        }

        cameraBridgeViewBase.setCameraPermissionGranted();
        cameraBridgeViewBase.enableView();
        verifTitle = findViewById(R.id.titleVerification);

        Button buttonYolo = (Button) findViewById(R.id.button2);
        buttonYolo.setOnClickListener(this::Yolo);

        templates = retrieveTemplates();
    }

    @Override
    public void onCameraViewStarted(int width, int height) {

    }

    @Override
    public void onCameraViewStopped() {

    }

    @Override
    public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {
        Mat frame = inputFrame.rgba();
        if (isNotMoving(frame.clone())) {
            System.out.println("EarRecognition: not moving");
            frame = performYoloDetection(frame);
        }

        return frame;
    }

    private Mat performYoloDetection(Mat frame) {
        if (startYolo){
            System.out.println("EarRecognition: yolo is processing");

            // rotate image since we use portrait mode
            int[] targetSize = {frame.size(1), frame.size(0)};
            Mat tImg = new Mat(targetSize, frame.type());
            Core.transpose(frame, tImg);
            Core.flip(tImg, frame, 0);

            Mat grayFrame = frame.clone();

            Imgproc.cvtColor(grayFrame, grayFrame, Imgproc.COLOR_RGBA2GRAY);
            Imgproc.cvtColor(grayFrame, grayFrame, Imgproc.COLOR_GRAY2RGB);

            Mat imageBlob = Dnn.blobFromImage(grayFrame,
                    1./(255),
                    new Size(640, 640),
                    new Scalar(0,0,0),
                    false // In case input is BGR, set it to true
            );

            tinyYolo.setInput(imageBlob);

            List<Mat> result = new ArrayList<Mat>(2);

            List<String> outBlobNames = new ArrayList<>();

            outBlobNames.add(0, "output");

            tinyYolo.forward(result, outBlobNames);
            Mat res = result.get(0);

            res = res.reshape(1, 25200);

            List<Integer> clsIds = new ArrayList<>();
            List<Float> confs = new ArrayList<>();
            List<Rect2d> rects = new ArrayList<>();

            float x_factor = ((float) frame.width() )/ (float) 640.0;
            float y_factor = ((float) frame.height())/ (float) 640.0;

            for (int j=0; j<25200;j++) {

                Mat row = res.row(j);

                //Mat classes_scores = row.colRange(5, res.cols());
                Core.MinMaxLocResult minMaxLocResult = Core.minMaxLoc(row.colRange(5, 7));

                float confidence = (float) (row.get(0,4)[0]);
                Point classIdPoint = minMaxLocResult.maxLoc;

                if (confidence > 0.85) {

                    int x = (int) (row.get(0,0)[0]);
                    int y = (int) (row.get(0,1)[0]);
                    int w   = (int) (row.get(0,2)[0]);
                    int h  = (int) (row.get(0,3)[0]);

                    int left = (int) ((x - 0.5 * w ) * x_factor);
                    int top = (int) ((y-0.5*h) * y_factor);
                    int width = (int) (w * x_factor);
                    int height = (int) (h * y_factor);

                    Rect roi = new Rect(left, top, width, height);
                    Mat crop = new Mat(frame, roi);
                    croppedEar = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888);
                    Utils.matToBitmap(crop, croppedEar);

                    clsIds.add((int)classIdPoint.x);
                    confs.add((float) confidence);
                    Rect2d box = new Rect2d(left, top, width, height);
                    rects.add(box);
                    System.out.println("Predicted " + classIdPoint.x + " confidence " + confidence);
                }
            }

            if (confs.size() > 0) {
                MatOfFloat confidences = new MatOfFloat(Converters.vector_float_to_Mat(confs));
                Rect2d[] boxesArray = rects.toArray(new Rect2d[0]);
                MatOfRect2d boxes = new MatOfRect2d();
                boxes.fromList(rects);
                MatOfInt indices = new MatOfInt();
                float nmsThresh = 0.2f;

                Dnn.NMSBoxes(boxes, confidences, (float)  0.25, (float) 0.45, indices);

                int[] ind = indices.toArray();
                float max_conf = -1;
                for (int i = 0; i <ind.length; i++) {
                    float conf = confs.get(ind[i]);
                    if (conf > max_conf) max_conf = conf;
                }

                for (int i = 0; i < ind.length; i++) {
                    int idx = ind[i];
                    Rect2d box = boxesArray[idx];
                    int idGuy = clsIds.get(idx);

                    float conf = confs.get(idx);
                    if (conf == max_conf) {

                        List<String> labelNames = Arrays.asList("LeftEar", "RightEar");
                        String intConf = new Integer((int) (conf * 100)).toString();
                        Imgproc.cvtColor(frame, frame, Imgproc.COLOR_RGBA2BGR);
                        Imgproc.putText(frame, labelNames.get(idGuy) + intConf + "%", box.tl(), Imgproc.FONT_HERSHEY_SIMPLEX, 1, new Scalar(255, 0, 240), 2);
                        Imgproc.rectangle(frame, box.br(), box.tl(), new Scalar(0, 255, 0), 2);
                        Imgproc.cvtColor(frame, frame, Imgproc.COLOR_BGR2RGBA);
                    }
                }
            }

            // rotate again
            targetSize[0] = frame.size(1);
            targetSize[1] = frame.size(0);

            tImg = new Mat(targetSize, frame.type());
            Core.transpose(frame, tImg);
            Core.flip(tImg, frame, 1);
        }
        return frame;
    }

    private void performVerification(){
        if(templates.size()<3){
            verifTitle.setText("Templates not loaded");
            return;
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

        double maxSimilarity = calculateSimilarity(probe, templates);
        if(maxSimilarity > THRESHOLD) {
            verificationSuccess = true;
            similarityAchieved = maxSimilarity;
            verifTitle.setText("Success, " + maxSimilarity);
        }
        else {
            verificationSuccess = false;
            verifTitle.setText("Failed, " + maxSimilarity);
        }
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

    private ArrayList<double[]> retrieveTemplates(){
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

    public void onTryButtonClick(View button){
        if(croppedEar!= null){
            handler.post(new Runnable() {
                @Override
                public void run() {
                    performVerification();
                }
            });
        }
    }

    @Override
    protected void onResume() {
        super.onResume();
    }

    @Override
    protected void onPause() {
        super.onPause();
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
    }
}