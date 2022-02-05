package com.biometricssystems.earrecognition;

import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;

import android.Manifest;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.os.Bundle;
import android.provider.MediaStore;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.Toast;

import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
import org.opencv.core.MatOfFloat;
import org.opencv.core.MatOfInt;
import org.opencv.core.MatOfRect2d;
import org.opencv.core.Point;
import org.opencv.core.Rect2d;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.dnn.Dnn;
import org.opencv.dnn.Net;
import org.opencv.imgproc.Imgproc;
import org.opencv.utils.Converters;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.Charset;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class NewIdentityActivity extends AppCompatActivity {

    final int REQUEST_CODE = 7;
    public static final int RequestPermissionCode = 1;

    Button btnTakePhoto;
    List<ImageView> imageViews = new ArrayList<>();
    List<Bitmap> imgBitmaps = new ArrayList<>();

    int imageCounter = 0;
    boolean imageCaptured = false;

    Net tinyYolo=null;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_new_identity);

        if(!OpenCVLoader.initDebug()){
            Log.d("CVerror","OpenCV library Init failure");
        }else{
            // load your library and do initializing stuffs like System.loadLibrary();
        }
        initYolo();

        btnTakePhoto = findViewById(R.id.btnTakePicture);
        imageViews.add(findViewById(R.id.imageOne));
        imageViews.add(findViewById(R.id.imageTwo));
        imageViews.add(findViewById(R.id.imageThree));

        // Initialize imgBitmaps with a size of 3
        for (int i = 0; i<3 ; i++) {
            imgBitmaps.add(null);
        }

        EnableRuntimePermission();

        btnTakePhoto.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                if (!imageCaptured){
                    Intent intent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
                    startActivityForResult(intent, REQUEST_CODE);
                } else {
                    submitImages();
                }
            }
        });

        for (int i = 0; i< imageViews.size(); i++) {
            int final_i = i + 1;
            imageViews.get(i).setOnClickListener(new View.OnClickListener() {
                @Override
                public void onClick(View view) {
                    if (imageCaptured) {
                        Intent intent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
                        startActivityForResult(intent, REQUEST_CODE + final_i);
                    }
                }
            });
        }
    }


    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (requestCode == REQUEST_CODE && resultCode == RESULT_OK) {
            if (imageCounter < 3) {
                Bitmap bitmap = (Bitmap) data.getExtras().get("data");
                localizeAndSegmentEar(bitmap);
                imageViews.get(imageCounter).setImageBitmap(bitmap);
                imgBitmaps.add(imageCounter, bitmap);

                 // Prepare to capture next image
                imageCounter++;
                if (imageCounter == 3) {
                    imageCaptured = true;
                    btnTakePhoto.setText("Register");
                    return;
                };

                Intent intent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
                startActivityForResult(intent, REQUEST_CODE);
            }
        }
        // Handle re-capture cases
        else if (requestCode >= REQUEST_CODE+1 && requestCode <= REQUEST_CODE+3 && resultCode == RESULT_OK) {
            Bitmap bitmap = (Bitmap) data.getExtras().get("data");
            int img_index = requestCode-(REQUEST_CODE+1);
            localizeAndSegmentEar(bitmap);
            imageViews.get(img_index).setImageBitmap(bitmap);
        }

    }

    private void localizeAndSegmentEar(Bitmap bitmap) {
        Mat frame = new Mat();
        Utils.bitmapToMat(bitmap, frame);
        Core.flip(frame, frame, 1);

        Mat grayFrame = frame.clone();

        Imgproc.cvtColor(grayFrame, grayFrame, Imgproc.COLOR_RGB2GRAY);
        Imgproc.cvtColor(grayFrame, grayFrame, Imgproc.COLOR_GRAY2RGB);

        int numcols = grayFrame.cols();
        int numrows = grayFrame.rows();
        int _max = Math.max(numcols, numrows);

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

                    List<String> labelNames = Arrays.asList("Left", "Right");
                    String intConf = new Integer((int) (conf * 100)).toString();
                    Imgproc.cvtColor(frame, frame, Imgproc.COLOR_RGB2BGR);
                    Imgproc.rectangle(frame, box.br(), box.tl(), new Scalar(0, 255, 0), 1);
                    Point textLoc = new Point(box.tl().x, box.tl().y-2);
                    Imgproc.putText(frame, labelNames.get(idGuy) + intConf + "%", textLoc, Imgproc.FONT_HERSHEY_SIMPLEX, 0.5, new Scalar(255, 0, 240), 1);
                    Imgproc.cvtColor(frame, frame, Imgproc.COLOR_BGR2RGB);
                }
            }
        }
        Utils.matToBitmap(frame, bitmap);
    }


    public void EnableRuntimePermission(){
        if (ActivityCompat.shouldShowRequestPermissionRationale(NewIdentityActivity.this,
                Manifest.permission.CAMERA)) {
            Toast.makeText(NewIdentityActivity.this,"CAMERA permission allows us to Access CAMERA app",     Toast.LENGTH_LONG).show();
        } else {
            ActivityCompat.requestPermissions(NewIdentityActivity.this,new String[]{
                    Manifest.permission.CAMERA}, RequestPermissionCode);
        }
    }
    @Override
    public void onRequestPermissionsResult(int requestCode, String permissions[], int[] result) {
        super.onRequestPermissionsResult(requestCode, permissions, result);
        switch (requestCode) {
            case RequestPermissionCode:
                if (result.length > 0 && result[0] == PackageManager.PERMISSION_GRANTED) {
                    Toast.makeText(NewIdentityActivity.this, "Permission Granted, Now your application can access CAMERA.", Toast.LENGTH_LONG).show();
                } else {
                    Toast.makeText(NewIdentityActivity.this, "Permission Canceled, Now your application cannot access CAMERA.", Toast.LENGTH_LONG).show();
                }
                break;
        }
    }

    void submitImages() {
        // Handle the pictures here
    }

    public void initYolo() {
        if(tinyYolo == null) {
            MatOfByte yoloWeights = new MatOfByte();
            try {
                yoloWeights.fromArray(loadTextFromAssets("ears.onnx", null));
            } catch (IOException e) {
                e.printStackTrace();
            }

            tinyYolo = Dnn.readNetFromONNX(yoloWeights);
            Toast.makeText(this, "EarRecognition: yolo initialized", Toast.LENGTH_SHORT).show();
        }
    }

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
}