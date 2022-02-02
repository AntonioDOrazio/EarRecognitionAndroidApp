package com.biometricssystems.earrecognition;

import androidx.appcompat.app.AppCompatActivity;

import android.content.Intent;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;

public class MainActivity extends AppCompatActivity {

    private Button btnNewIdentity;
    private Button btnVerification;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        Button btnNewIdentity = (Button) findViewById(R.id.btnNewIdentity);
        Button btnVerification = (Button) findViewById(R.id.btnVerification);

        btnNewIdentity.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                Intent i = new Intent(MainActivity.this, NewIdentityActivity.class);
                MainActivity.this.startActivity(i);
            }
        });

        btnVerification.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                Intent i = new Intent(MainActivity.this, VerificationActivity.class);
                MainActivity.this.startActivity(i);
            }
        });
    }
}