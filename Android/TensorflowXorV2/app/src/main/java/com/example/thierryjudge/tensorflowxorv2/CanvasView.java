package com.example.thierryjudge.tensorflowxorv2;

import android.app.Activity;
import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.Path;
import android.graphics.Rect;
import android.support.annotation.Nullable;
import android.util.AttributeSet;
import android.util.Log;
import android.view.MotionEvent;
import android.view.View;

import java.util.ArrayList;

public class CanvasView extends View
{
    private Bitmap bitmap;
    private Paint paint;

    MainActivity listener = null;

    int paintColor = Color.BLACK;
    float paintWidth = 10f;

    Context context;

    int pointColor = Color.RED;

    float mx, my;

    public CanvasView(Context context, @Nullable AttributeSet attrs) {
        super(context, attrs);
        this.context = context;

        paint = new Paint();
        setPaintSettings();
    }

    public void setListener(MainActivity listener)
    {
        this.listener = listener;
    }

    private void setPaintSettings()
    {
        paint.setAntiAlias(true);
        paint.setColor(paintColor);
        paint.setStyle(Paint.Style.STROKE);
        paint.setStrokeJoin(Paint.Join.ROUND);
        paint.setStrokeWidth(paintWidth);
        paint.setStrokeCap(Paint.Cap.ROUND);
    }

    @Override
    protected void onSizeChanged(int w, int h, int oldw, int oldh)
    {
        super.onSizeChanged(w, h, oldw, oldh);
        bitmap = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888);
    }


    private void onStartTouch(float x, float y)
    {
        int width = bitmap.getWidth();
        int height = bitmap.getHeight();
        Log.d("TEST", x + ", " + y);

        mx = x;
        my = y;

        x = (x - (width/2)) / (width/2);
        y = - (y - (height/2)) / (height/2);

        Log.d("TEST", x + ", " + y);
        int prediction = listener.touch(x,y);

        if(prediction == 1)
        {
            pointColor = Color.RED;
        }
        else
        {
            pointColor = Color.BLUE;
        }
    }


    @Override
    public boolean onTouchEvent(MotionEvent event) {
        float x = event.getX();
        float y = event.getY();

        switch (event.getAction()) {
            case MotionEvent.ACTION_DOWN:
                onStartTouch(x, y);
                invalidate();
                break;
        }


        return true;
    }

    protected void onDraw(Canvas canvas)
    {
        int width = bitmap.getWidth();
        int height = bitmap.getHeight();
        canvas.drawRect(new Rect(0, 0, width, height), paint);
        canvas.drawLine(width/2, 0, width/2, height, paint);
        canvas.drawLine(0, height/2, width, height/2, paint);

        Paint newPaint = new Paint(paint);
        newPaint.setColor(pointColor);
        canvas.drawCircle(mx, my, 10, newPaint);
    }

}
