package com.example.thierryjudge.tensorflowxor;

import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.NumberPicker;
import android.widget.TextView;

import org.tensorflow.contrib.android.TensorFlowInferenceInterface;

public class MainActivity extends AppCompatActivity {

    NumberPicker numberPicker1 = null;
    NumberPicker numberPicker2 = null;
    Button button = null;
    TextView textView = null;

    TensorFlowInferenceInterface inferenceInterface = null;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        inferenceInterface = new TensorFlowInferenceInterface(getAssets(), "frozen_xor_graph.pb");

        numberPicker1 = (NumberPicker) findViewById(R.id.numberPicker);
        numberPicker1.setMinValue(0);
        numberPicker1.setMaxValue(1);
        numberPicker1.setWrapSelectorWheel(true);

        numberPicker2 = (NumberPicker) findViewById(R.id.numberPicker2);
        numberPicker2.setMinValue(0);
        numberPicker2.setMaxValue(1);
        numberPicker2.setWrapSelectorWheel(true);

        textView = (TextView) findViewById(R.id.textView);

        button = (Button) findViewById(R.id.button);
        button.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                predict();
            }
        });
    }


    private void predict()
    {
        float[] output = new float[1];
        float[] input = {numberPicker1.getValue(), numberPicker2.getValue()};

        inferenceInterface.feed("input",input, 1, 2);
        inferenceInterface.run(new String[]{"output"});
        inferenceInterface.fetch("output", output);


        textView.setText(output[0] + "");
    }
}
