package com.lanytek.deepsensev3;

import android.app.Activity;
import android.content.Intent;
import android.database.Cursor;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.net.Uri;
import android.os.AsyncTask;
import android.os.Environment;
import android.provider.MediaStore;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;

import com.squareup.picasso.Picasso;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

public class MainActivity extends AppCompatActivity {
    public static String TAG = "DeepSense";

    private List<String> img_recognition_descriptions = new ArrayList<>();
    private static final String [] yolo_descriptions = {"aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"};

    private static String model_yolo_tiny = (new File(Environment.getExternalStorageDirectory(), "YoloModels/Yolo-Tiny-New-Format")).getAbsolutePath();
    private static String model_img_recognition = (new File(Environment.getExternalStorageDirectory(), "ImageNetModels/Vgg_F-New-Format")).getAbsolutePath();

    private Activity activity = this;

    private ImageView iv;

    private Button btn_loadModelGPU;
    private Button btn_processImage;
    private TextView tv_runtime, tv_desc;

    private static final int SELECT_PICTURE = 9999;
    private String selectedImagePath = null;

    // Used to load the 'native-lib' library on application startup.
    static {
        System.loadLibrary("deepsense");
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        iv = (ImageView) findViewById(R.id.iv_image);
        btn_loadModelGPU = (Button) findViewById(R.id.btn_loadModelGPU);
        btn_processImage = (Button) findViewById(R.id.btn_processImage);
        tv_runtime = (TextView) findViewById(R.id.tv_runTime);
        tv_desc = (TextView) findViewById(R.id.tv_desc);

        new async_copy_kernel_code().execute("deepsense.cl");

        btn_loadModelGPU.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                new async_loadModel().execute();
            }
        });

        btn_processImage.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                new async_processImage_yolo().execute();
                //new async_processImage_img_recognition().execute();
            }
        });

        iv.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent intent = new Intent();
                intent.setType("image/*");
                intent.setAction(Intent.ACTION_GET_CONTENT);
                startActivityForResult(Intent.createChooser(intent, "Select Picture"), SELECT_PICTURE);
            }
        });
    }

    private void setButtons(boolean isEnabled) {
        //btn_loadModelCPU.setEnabled(isEnabled);
        btn_loadModelGPU.setEnabled(isEnabled);
        btn_processImage.setEnabled(isEnabled);
    }

    private class async_copy_kernel_code extends AsyncTask<String, Void, Void> {

        @Override
        protected void onPreExecute() {
            super.onPreExecute();
            setButtons(false);
        }

        @Override
        protected Void doInBackground(String... params) {
            for(String p : params) {
                Utilities.copyFile(activity, p);
            }
            return null;
        }

        @Override
        protected void onPostExecute(Void aVoid) {
            super.onPostExecute(aVoid);
            setButtons(true);
        }
    }

    private class async_loadModel extends AsyncTask<Void, Void, Void> {

        @Override
        protected void onPreExecute() {
            setButtons(false);
            super.onPreExecute();
        }

        @Override
        protected Void doInBackground(Void... params) {
            if(new File(model_img_recognition + "/description").exists()) {
                try {
                    img_recognition_descriptions.clear();
                    BufferedReader br = new BufferedReader(new FileReader(new File(model_img_recognition + "/description")));
                    String line;
                    while((line = br.readLine()) != null) {
                        img_recognition_descriptions.add(line);
                    }
                    br.close();
                } catch (FileNotFoundException e) {
                    e.printStackTrace();
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }

            InitGPU(model_yolo_tiny, activity.getPackageName());
            return null;
        }

        @Override
        protected void onPostExecute(Void aVoid) {
            super.onPostExecute(aVoid);
            setButtons(true);
        }
    }

    private class async_processImage_img_recognition extends AsyncTask<Void, Void, Void> {

        private double t1,t2;
        private double cnn_runtime;
        private float [] result;
        private Bitmap bm = null;
        private int best_idx = -1;

        @Override
        protected void onPreExecute() {
            btn_processImage.setEnabled(false);
            tv_runtime.setText("------");
            tv_desc.setText("...");
            t1 = System.currentTimeMillis();
            super.onPreExecute();
        }

        @Override
        protected void onPostExecute(Void aVoid) {
            super.onPostExecute(aVoid);
            t2 = System.currentTimeMillis();
            double runtime = t2 - t1;
            btn_processImage.setEnabled(true);
            tv_runtime.setText(cnn_runtime + " / " + runtime + " ms");
            tv_desc.setText(img_recognition_descriptions.get(best_idx));
        }

        @Override
        protected Void doInBackground(Void... voids) {

            if(selectedImagePath != null) {
                final int IMG_X = 224;
                final int IMG_Y = 224;
                final int IMG_C = 3;

                final float [] bitmapArray = new float[IMG_X * IMG_Y * IMG_C];

                try {
                    bm = Picasso.with(activity)
                            .load(new File(selectedImagePath))
                            .config(Bitmap.Config.ARGB_8888)
                            .resize(448,448)
                            .get();
                } catch (IOException e) {
                    e.printStackTrace();
                }

                if(bm != null) {
                    ExecutorService executor = Executors.newFixedThreadPool(8);

                    final double scaleX = (double)IMG_X / (double)bm.getWidth();
                    final double scaleY = (double)IMG_Y / (double)bm.getHeight();

                    for(int i = 0 ; i < 224 ; i++) {
                        final int finalI = i;
                        executor.execute(new Runnable() {
                            @Override
                            public void run() {
                                for(int j = 0 ; j < IMG_Y ; j++) {
                                    int pixel = bm.getPixel((int)Math.ceil(1/scaleX * finalI),(int)Math.ceil(1/scaleY * j));
                                    float b = (float)(pixel & 0x000000ff);
                                    float g = (float)((pixel >> 8) & 0x000000ff);
                                    float r = (float)((pixel >> 16) & 0x000000ff);
                                    int index = finalI * IMG_Y + j;
                                    bitmapArray[index * 3] = r - 122.803f;
                                    bitmapArray[index * 3 + 1] = g - 114.885f;
                                    bitmapArray[index * 3 + 2] = b - 101.572f;
                                }
                            }
                        });
                    }

                    executor.shutdown();
                    try {
                        executor.awaitTermination(Long.MAX_VALUE, TimeUnit.MILLISECONDS);
                    } catch (InterruptedException e) {
                        e.printStackTrace();
                    }

                    double x1 = System.currentTimeMillis();
                    float [] result = GetInferrence(bitmapArray);
                    double x2 = System.currentTimeMillis();
                    cnn_runtime = x2 - x1;
                    Log.d(TAG,"CNN RUNTIME: " + cnn_runtime + "ms");

                    //get top-1
                    float best_prob = 0;
                    for(int i = 0 ; i < 1000 ; i ++) {
                        if(best_prob < result[i]) {
                            best_idx = i;
                            best_prob = result[i];
                        }
                    }

                    Log.d(TAG,"Image classified as : " + img_recognition_descriptions.get(best_idx));
                }
            }

            return null;
        }
    }

    private class async_processImage_yolo extends AsyncTask<Void, Void, Void> {

        private double t1,t2;
        private double cnn_runtime;
        private float [] result;
        private Bitmap bm = null;

        @Override
        protected void onPreExecute() {
            btn_processImage.setEnabled(false);
            tv_runtime.setText("------");
            t1 = System.currentTimeMillis();
            super.onPreExecute();
        }

        @Override
        protected Void doInBackground(Void... params) {

            if(selectedImagePath != null) {
                final int IMG_X = 448;
                final int IMG_Y = 448;
                final int IMG_C = 3;

                final float [] bitmapArray = new float[IMG_X * IMG_Y * IMG_C];

                try {
                    bm = Picasso.with(activity)
                            .load(new File(selectedImagePath))
                            .config(Bitmap.Config.ARGB_8888)
                            .resize(IMG_X,IMG_Y)
                            .get();
                } catch (IOException e) {
                    e.printStackTrace();
                }

                if(bm != null) {
                    /*ExecutorService executor = Executors.newFixedThreadPool(8);

                    for(int w = 0 ; w < bm.getWidth() ; w++) {
                        final int finalW = w;
                        executor.execute(new Runnable() {
                            @Override
                            public void run() {
                                for(int h = 0 ; h < bm.getHeight() ; h++) {
                                    int pixel = bm.getPixel(finalW, h);
                                    for(int c = 0 ; c < 3 ; c++) {
                                        bitmapArray[h * IMG_X * IMG_C + finalW * IMG_C + c] = getColorPixel(pixel, c);
                                    }
                                }
                            }
                        });
                    }

                    executor.shutdown();
                    try {
                        executor.awaitTermination(Long.MAX_VALUE, TimeUnit.MILLISECONDS);
                    } catch (InterruptedException e) {
                        e.printStackTrace();
                    }*/

                    for(int w = 0 ; w < bm.getWidth() ; w++) {
                        for(int h = 0 ; h < bm.getHeight() ; h++) {
                            int pixel = bm.getPixel(w, h);
                            for(int c = 0 ; c < 3 ; c++) {
                                bitmapArray[h * IMG_X * IMG_C + w * IMG_C + c] = Utilities.getColorPixel(pixel, c);
                            }
                        }
                    }
                }

                double x1 = System.currentTimeMillis();
                float [] result = GetInferrence(bitmapArray);
                double x2 = System.currentTimeMillis();
                cnn_runtime = x2 - x1;
                Log.d(TAG,"CNN RUNTIME: " + cnn_runtime + "ms");

                int classes = 20;
                int side = 7;
                int num = 2;
                float thresh = 0.15f;

                //process result first
                float [][] probs = new float[side * side * num][classes];
                Utilities.box[] boxes = new Utilities.box[side * side * num];
                for(int j = 0 ; j < boxes.length ; j++)
                    boxes[j] = new Utilities.box();

                Utilities.convert_yolo_detections(result, classes, num, 1, side, 1, 1, thresh, probs, boxes, 0);
                Utilities.do_nms_sort(boxes, probs, side * side * num, classes, 0.5f);

                //do box drawing
                final Bitmap mutableBitmap = Bitmap.createScaledBitmap(
                        bm, 512, 512, false).copy(bm.getConfig(), true);
                final Canvas canvas = new Canvas(mutableBitmap);

                for(int i = 0; i < side * side * num; ++i){

                    int classid = -1;
                    float maxprob = -100000.0f;
                    for(int j = 0 ; j < classes ; j++) {
                        if(probs[i][j] > maxprob) {
                            classid = j;
                            maxprob = probs[i][j];
                        }
                    }

                    if(classid < 0)
                        continue;

                    float prob = probs[i][classid];
                    if(prob > thresh){
                        Utilities.box b = boxes[i];

                        int left  = (int) ((b.x-b.w/2.) * mutableBitmap.getWidth());
                        int right = (int) ((b.x+b.w/2.) * mutableBitmap.getWidth());
                        int top   = (int) ((b.y-b.h/2.) * mutableBitmap.getHeight());
                        int bot   = (int) ((b.y+b.h/2.) * mutableBitmap.getHeight());

                        if(left < 0) left = 0;
                        if(right > mutableBitmap.getWidth() - 1) right = mutableBitmap.getWidth() - 1;
                        if(top < 0) top = 0;
                        if(bot > mutableBitmap.getHeight() - 1) bot = mutableBitmap.getHeight() - 1;

                        Paint p = new Paint();
                        p.setStrokeWidth(p.getStrokeWidth() * 3);
                        p.setColor(Color.RED);
                        canvas.drawLine(left, top, right, top, p);
                        canvas.drawLine(left, top, left, bot, p);
                        canvas.drawLine(left, bot, right, bot, p);
                        canvas.drawLine(right, top, right, bot, p);

                        p.setTextSize(48f);
                        p.setColor(Color.BLUE);
                        canvas.drawText("" + yolo_descriptions[classid],left + (right - left)/2,top + (bot - top)/2,p);
                    }
                }

                activity.runOnUiThread(new Runnable() {
                    @Override
                    public void run() {
                        iv.setImageBitmap(mutableBitmap);
                    }
                });
            }

            return null;
        }

        @Override
        protected void onPostExecute(Void aVoid) {
            super.onPostExecute(aVoid);
            t2 = System.currentTimeMillis();
            double runtime = t2 - t1;
            btn_processImage.setEnabled(true);
            tv_runtime.setText(cnn_runtime + " / " + runtime + " ms");
        }
    }

    public String getPath(Uri uri) {
        String[] projection = { MediaStore.Images.Media.DATA };
        Cursor cursor = managedQuery(uri, projection, null, null, null);
        int column_index = cursor.getColumnIndexOrThrow(MediaStore.Images.Media.DATA);
        cursor.moveToFirst();
        return cursor.getString(column_index);
    }

    public void onActivityResult(int requestCode, int resultCode, Intent data) {
        if (resultCode == RESULT_OK) {
            if (requestCode == SELECT_PICTURE) {
                Uri selectedImageUri = data.getData();
                selectedImagePath = getPath(selectedImageUri);
                if(selectedImagePath != null)
                    iv.setImageURI(selectedImageUri);
            }
        }
    }

    /**
     * A native method that is implemented by the 'native-lib' native library,
     * which is packaged with this application.
     */
    public native void InitGPU(String model_dir_path, String packageName);
    public native float [] GetInferrence(float [] input);

}
