package com.biometricssystems.earrecognition;

import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;

import android.Manifest;
import android.content.ContentValues;
import android.content.Context;
import android.content.Intent;
import android.content.SharedPreferences;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.Matrix;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.Toast;

import com.biometricssystems.earrecognition.ml.Model;
import com.biometricssystems.earrecognition.models.EarRecognition;
import com.biometricssystems.earrecognition.models.YoloDetection;

import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
import org.opencv.core.MatOfDouble;
import org.opencv.core.MatOfFloat;
import org.opencv.core.MatOfInt;
import org.opencv.core.MatOfRect2d;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Rect2d;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.dnn.Dnn;
import org.opencv.dnn.Net;
import org.opencv.imgproc.Imgproc;
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

public class NewIdentityActivity extends AppCompatActivity {

    final int REQUEST_CODE = 7;
    public static final int RequestPermissionCode = 1;

    Button btnTakePhoto;
    List<ImageView> imageViews = new ArrayList<>();
    List<Bitmap> imgBitmaps = new ArrayList<>(3);
    List<Bitmap> imgCropped = new ArrayList<>(3);
    ArrayList<Uri> imageUris = new ArrayList<>(3);

    int imageCounter = 0;
    boolean imageCaptured = false;

    YoloDetection yolo;
    EarRecognition recognition;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_new_identity);

        if(!OpenCVLoader.initDebug()){
            Log.d("CVerror","OpenCV library Init failure");
        }else{
            // load your library and do initializing stuffs like System.loadLibrary();
        }
        yolo = new YoloDetection(this);
        recognition = new EarRecognition(this, null);

        btnTakePhoto = findViewById(R.id.btnTakePicture);
        imageViews.add(findViewById(R.id.imageOne));
        imageViews.add(findViewById(R.id.imageTwo));
        imageViews.add(findViewById(R.id.imageThree));

        EnableRuntimePermission();

        btnTakePhoto.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                if (!imageCaptured){
                    ContentValues values = new ContentValues();
                    values.put(MediaStore.Images.Media.TITLE, "New Picture");
                    values.put(MediaStore.Images.Media.DESCRIPTION, "From your Camera");
                    Uri uri = getContentResolver().insert(
                            MediaStore.Images.Media.EXTERNAL_CONTENT_URI, values);
                    imageUris.add(uri);
                    Intent intent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);

                    intent.putExtra(MediaStore.EXTRA_OUTPUT, imageUris.get(imageCounter));
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
                        int PICTURE_RESULT = REQUEST_CODE + final_i;

                        ContentValues values = new ContentValues();
                        values.put(MediaStore.Images.Media.TITLE, "Picture");
                        values.put(MediaStore.Images.Media.DESCRIPTION, "From your Camera");
                        Uri uri = getContentResolver().insert(
                                MediaStore.Images.Media.EXTERNAL_CONTENT_URI, values);
                        imageUris.set(final_i-1, uri);
                        Intent intent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
                        intent.putExtra(MediaStore.EXTRA_OUTPUT, imageUris.get(final_i-1));
                        startActivityForResult(intent, PICTURE_RESULT);
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
                try {
                    Uri uri = imageUris.get(imageCounter);
                    Bitmap bitmap = MediaStore.Images.Media.getBitmap(
                            getContentResolver(), uri);

                    Mat frame = new Mat();
                    Utils.bitmapToMat(bitmap, frame);
                    Core.flip(frame, frame, 0);
                    frame = yolo.localizeAndSegmentEar(frame, false, true);
                    Utils.matToBitmap(frame, bitmap);

                    Matrix matrix = new Matrix();
                    matrix.postRotate(-90);
                    bitmap = Bitmap.createBitmap(bitmap, 0, 0, bitmap.getWidth(), bitmap.getHeight(), matrix, true);

                    Bitmap croppedImage = yolo.getCroppedEar();

                    if(croppedImage == null){
                        Toast.makeText(this, "EarRecognition: ear not found", Toast.LENGTH_SHORT).show();
                    }

                    imageViews.get(imageCounter).setImageBitmap(bitmap);
                    imgBitmaps.add(imageCounter, bitmap);
                    imgCropped.add(imageCounter, croppedImage);

                    // Prepare to capture next image
                    imageCounter++;
                    if (imageCounter == 3) {
                        imageCaptured = true;
                        btnTakePhoto.setText("Register");
                        return;
                    };
                    ContentValues values = new ContentValues();
                    values.put(MediaStore.Images.Media.TITLE, "Picture");
                    values.put(MediaStore.Images.Media.DESCRIPTION, "From your Camera");
                    imageUris.add(getContentResolver().insert(
                            MediaStore.Images.Media.EXTERNAL_CONTENT_URI, values));

                    Intent intent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
                    intent.putExtra(MediaStore.EXTRA_OUTPUT, imageUris.get(imageCounter));
                    startActivityForResult(intent, REQUEST_CODE);

                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
        }
        // Handle re-capture cases
        else if (requestCode >= REQUEST_CODE+1 && requestCode <= REQUEST_CODE+3 && resultCode == RESULT_OK) {
            try {
                int img_index = requestCode-(REQUEST_CODE+1);
                Bitmap bitmap = MediaStore.Images.Media.getBitmap(
                        getContentResolver(), imageUris.get(img_index));

                Mat frame = new Mat();
                Utils.bitmapToMat(bitmap, frame);
                Core.flip(frame, frame, 0);
                frame = yolo.localizeAndSegmentEar(frame, false, true);
                Utils.matToBitmap(frame, bitmap);

                Matrix matrix = new Matrix();
                matrix.postRotate(-90);
                bitmap = Bitmap.createBitmap(bitmap, 0, 0, bitmap.getWidth(), bitmap.getHeight(), matrix, true);

                Bitmap croppedImage = yolo.getCroppedEar();

                if(croppedImage == null){
                    Toast.makeText(this, "EarRecognition: ear not found", Toast.LENGTH_SHORT).show();
                }
                imageViews.get(img_index).setImageBitmap(bitmap);
                imgCropped.set(img_index, croppedImage);
            } catch (IOException e) {
                e.printStackTrace();
            }
        }

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
        for (int i=0; i<imgCropped.size(); i++) {
            recognition.extractAndSaveFeatures(imgCropped.get(i), i, getSharedPreferences(getString(R.string.app_name), MODE_PRIVATE));
        }
        Toast.makeText(this, "Saved Templates!", Toast.LENGTH_LONG).show();
    }
}