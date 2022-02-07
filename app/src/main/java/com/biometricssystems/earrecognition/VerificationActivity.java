package com.biometricssystems.earrecognition;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;

import android.Manifest;
import android.content.Context;
import android.content.SharedPreferences;
import android.content.pm.PackageManager;
import android.content.res.Resources;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Color;
import android.os.Bundle;
import android.os.Handler;
import android.util.Log;
import android.view.SurfaceView;
import android.view.View;
import android.widget.Button;
import android.widget.TextView;
import android.widget.Toast;

import com.biometricssystems.earrecognition.ml.Model;
import com.biometricssystems.earrecognition.models.EarRecognition;
import com.biometricssystems.earrecognition.models.YoloDetection;

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
    boolean frozenCapture;
    YoloDetection yolo;

    EarRecognition recognition;

    Mat oldFrame; // Initialization is later in the code
    Mat frozenFrame;

    TextView verifTitle;
    Button yoloBtn;

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
            frozenCapture = false;
            yoloBtn.setText("Stop");
            if(yolo == null)
                yolo = new YoloDetection(this);
        } else {
            startYolo = false;
            yoloBtn.setText("Start");
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
        yoloBtn = (Button) findViewById(R.id.button2);

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

        yoloBtn.setOnClickListener(this::Yolo);
        recognition = new EarRecognition(this, getSharedPreferences(getString(R.string.app_name), Context.MODE_PRIVATE));

        frozenCapture = false;
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
        if (!frozenCapture && startYolo && isNotMoving(frame.clone())) {
            System.out.println("EarRecognition: not moving");
            frame = yolo.localizeAndSegmentEar(frame, false, false);
            frozenCapture = yolo.isEarDetected();
            if(frozenCapture) {
                startYolo = false;
                frozenFrame = frame.clone();
                handler.post(new Runnable() {
                    @Override
                    public void run() {
                        yoloBtn.setText("Repeat");
                    }
                });
            }
        }
        if(frozenCapture) {
            Bitmap temp = Bitmap.createBitmap(frame.width(), frame.height(), Bitmap.Config.ARGB_8888);
            Utils.matToBitmap(frozenFrame, temp);
            return frozenFrame;
        }
        else
            return frame;
    }

    public void onTryButtonClick(View button){
        if(yolo.isEarDetected()){
            handler.post(new Runnable() {
                @Override
                public void run() {
                    if(recognition.performVerification(yolo.getCroppedEar())){
                        // performed verification
                        if(recognition.isVerificationSuccess()) {
                            verifTitle.setText("Success, " + recognition.getSimilarityAchieved());
                            verifTitle.setTextColor(getColor(R.color.success));
                        }
                        else {
                            verifTitle.setText("Failed, " + recognition.getSimilarityAchieved());
                            verifTitle.setTextColor(getColor(R.color.failure));
                        }
                    } else {

                    }
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