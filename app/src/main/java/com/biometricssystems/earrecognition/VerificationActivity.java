package com.biometricssystems.earrecognition;

import android.Manifest;
import android.content.Context;
import android.content.SharedPreferences;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.os.Bundle;
import android.os.Handler;
import android.os.VibrationEffect;
import android.os.Vibrator;
import android.util.Log;
import android.view.SurfaceView;
import android.view.View;
import android.widget.Button;
import android.widget.TextView;
import android.widget.Toast;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;

import com.biometricssystems.earrecognition.models.EarRecognition;
import com.biometricssystems.earrecognition.models.YoloDetection;

import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.JavaCameraView;
import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.List;

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
        yoloBtn.performClick();
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
            frame = yolo.localizeAndSegmentEar(frame, true, false, false);
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
        final Vibrator vibrator = (Vibrator) getSystemService(Context.VIBRATOR_SERVICE);
        final VibrationEffect vibrationEffect = VibrationEffect.createOneShot(1000, VibrationEffect.DEFAULT_AMPLITUDE);

        if(yolo.isEarDetected()){
            handler.post(new Runnable() {
                @Override
                public void run() {
                    if(recognition.performVerification(yolo.getCroppedEar())){
                        // performed verification
                        if(recognition.isVerificationSuccess()) {
                            verifTitle.setText("Success, " + recognition.getSimilarityAchieved());
                            verifTitle.setTextColor(getColor(R.color.success));

                            // this is the only type of the vibration which requires system version Oreo (API 26)
                            if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.O) {
                                vibrator.cancel();
                                vibrator.vibrate(vibrationEffect);
                            }

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