package com.example.thierryjudge.tensorflowxorv2;

import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.widget.TextView;

import org.tensorflow.contrib.android.TensorFlowInferenceInterface;

public class MainActivity extends AppCompatActivity {

    private TextView textView = null;
    private TextView textView2 = null;

    TensorFlowInferenceInterface inferenceInterface = null;


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        CanvasView canvasView = (CanvasView) findViewById(R.id.canvas);
        textView = (TextView) findViewById(R.id.textview);
        textView2 = (TextView) findViewById(R.id.textview2);


        canvasView.setListener(this);

        inferenceInterface = new TensorFlowInferenceInterface(getAssets(), "frozen_xor_graph.pb");

    }

    public int touch(float x, float y)
    {
        float[] output = new float[1];
        float[] input = {x, y};

        inferenceInterface.feed("input",input, 1, 2);
        inferenceInterface.run(new String[]{"output"});
        inferenceInterface.fetch("output", output);

        int prediction = Math.round(output[0]);
        int real = 1;

        if ((x < 0 && y < 0) || (x >= 0 && y >= 0))
        {
            real = 0;
        }

        textView.setText("Prediction: " + prediction + ", Real: " + real);
        textView2.setText("Y: " + output[0]);

        return prediction;
    }
}
