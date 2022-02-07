package com.biometricssystems.earrecognition.models;

import android.content.Context;
import android.graphics.Bitmap;
import android.widget.Toast;

import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
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

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.Charset;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class YoloDetection {
    Net tinyYolo;
    Bitmap croppedEar;
    boolean earDetected = false;

    public YoloDetection(Context context){
        initYolo(context);
    }

    public Bitmap getCroppedEar(){
        return croppedEar;
    }

    public boolean isEarDetected() {
        return earDetected;
    }

    public Mat localizeAndSegmentEar(Mat frame, boolean rotate, boolean flipHorizontal, boolean highRes) {
        earDetected = false;
        if(rotate) {
            Core.transpose(frame, frame);
            Core.flip(frame, frame, 0);
        }

        Mat grayFrame = frame.clone();

        Imgproc.cvtColor(grayFrame, grayFrame, Imgproc.COLOR_RGBA2GRAY);
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
                    // ear detected
                    earDetected = true;
                    Rect roi = new Rect(box.br(), box.tl());
                    Mat crop = new Mat(frame, roi);
                    croppedEar = Bitmap.createBitmap((int)box.width, (int)box.height, Bitmap.Config.ARGB_8888);
                    Utils.matToBitmap(crop, croppedEar);

                    // label ear on frame
                    List<String> labelNames;
                    if(flipHorizontal)
                        labelNames = Arrays.asList("Right", "Left");
                    else
                        labelNames = Arrays.asList("Left", "Right");

                    String intConf = new Integer((int) (conf * 100)).toString();

                    Imgproc.cvtColor(frame, frame, Imgproc.COLOR_RGB2BGR);
                    if(highRes) {
                        Imgproc.rectangle(frame, box.br(), box.tl(), new Scalar(0, 255, 0), 10);
                        Point textLoc = new Point(box.tl().x, box.tl().y - 25);
                        Imgproc.putText(frame, labelNames.get(idGuy) + intConf + "%", textLoc, Imgproc.FONT_HERSHEY_SIMPLEX, 10, new Scalar(255, 0, 240), 10);
                    } else {
                        Imgproc.rectangle(frame, box.br(), box.tl(), new Scalar(0, 255, 0), 2);
                        Point textLoc = new Point(box.tl().x, box.tl().y - 10);
                        Imgproc.putText(frame, labelNames.get(idGuy) + intConf + "%", textLoc, Imgproc.FONT_HERSHEY_SIMPLEX, 1, new Scalar(255, 0, 240), 2);
                    }

                    Imgproc.cvtColor(frame, frame, Imgproc.COLOR_BGR2RGBA);
                }
            }
        }

        if(rotate) {
            Core.flip(frame, frame, 0);
            Core.transpose(frame, frame);
        }

        return frame;
    }

    private void initYolo(Context context) {
        if(tinyYolo == null) {
            MatOfByte yoloWeights = new MatOfByte();
            try {
                yoloWeights.fromArray(loadTextFromAssets("ears.onnx", null, context));
            } catch (IOException e) {
                e.printStackTrace();
            }

            tinyYolo = Dnn.readNetFromONNX(yoloWeights);
            Toast.makeText(context, "EarRecognition: yolo initialized", Toast.LENGTH_SHORT).show();
        }
    }

    private byte[] loadTextFromAssets(String assetsPath, Charset charset, Context context) throws IOException {
        InputStream is = context.getAssets().open(assetsPath);
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
