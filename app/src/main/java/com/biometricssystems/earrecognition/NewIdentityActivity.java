package com.biometricssystems.earrecognition;

import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;

import android.Manifest;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.os.Bundle;
import android.provider.MediaStore;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.Toast;

import java.util.ArrayList;
import java.util.List;

public class NewIdentityActivity extends AppCompatActivity {

    final int REQUEST_CODE = 7;
    public static final int RequestPermissionCode = 1;

    Button btnTakePhoto;
    List<ImageView> imageViews = new ArrayList<>();
    List<Bitmap> imgBitmaps = new ArrayList<>();

    int imageCounter = 0;
    boolean imageCaptured = false;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_new_identity);

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
            int final_i = i+1;
            imageViews.get(i).setOnClickListener(new View.OnClickListener() {
                @Override
                public void onClick(View view) {
                    if(imageCaptured) {
                        Intent intent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
                        startActivityForResult(intent, REQUEST_CODE+final_i);
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
            imageViews.get(img_index).setImageBitmap(bitmap);
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
    }
}