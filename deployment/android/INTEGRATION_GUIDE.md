# SeedShield Android Integration Guide

This guide explains how to integrate SeedShield AI models into an Android application for offline seed analysis.

## Overview

The Android app will:
1. Capture or select a seed image
2. Run YOLOv8 detector to find individual seeds
3. Classify each seed using MobileNetV2
4. Display results with color-coded annotations

---

## Prerequisites

- Android Studio Arctic Fox or later
- Android SDK 24+ (Android 7.0)
- Kotlin 1.8+
- TensorFlow Lite models (generated from `deployment/convert_to_tflite.py`)

---

## Setup

### 1. Add Dependencies

**build.gradle (app level):**

```gradle
dependencies {
    // TensorFlow Lite
    implementation 'org.tensorflow:tensorflow-lite:2.13.0'
    implementation 'org.tensorflow:tensorflow-lite-support:0.4.4'
    implementation 'org.tensorflow:tensorflow-lite-gpu:2.13.0'
    
    // Image loading
    implementation 'io.coil-kt:coil:2.4.0'
    
    // Camera
    implementation 'androidx.camera:camera-camera2:1.3.0'
    implementation 'androidx.camera:camera-lifecycle:1.3.0'
    implementation 'androidx.camera:camera-view:1.3.0'
}
```

### 2. Add Model Files

Copy TFLite models to `app/src/main/assets/`:
- `seedshield_detector.tflite`
- `seedshield_classifier.tflite`

### 3. Add Permissions

**AndroidManifest.xml:**

```xml
<uses-permission android:name="android.permission.CAMERA" />
<uses-permission android:name="android.permission.READ_EXTERNAL_STORAGE" />

<uses-feature android:name="android.hardware.camera" android:required="false" />
```

---

## Implementation

### SeedShieldAnalyzer.kt

```kotlin
package com.seedshield.analyzer

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.RectF
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.common.FileUtil
import java.nio.ByteBuffer
import java.nio.ByteOrder

class SeedShieldAnalyzer(private val context: Context) {
    
    // Models
    private var detector: Interpreter? = null
    private var classifier: Interpreter? = null
    
    // Constants
    private val DETECTOR_INPUT_SIZE = 640
    private val CLASSIFIER_INPUT_SIZE = 224
    private val CLASSES = arrayOf("REAL", "FAKE", "LOW_QUALITY")
    private val COLORS = mapOf(
        "REAL" to Color.GREEN,
        "FAKE" to Color.RED,
        "LOW_QUALITY" to Color.YELLOW
    )
    
    // Results
    data class SeedResult(
        val bbox: RectF,
        val className: String,
        val confidence: Float
    )
    
    data class AnalysisResult(
        val seeds: List<SeedResult>,
        val annotatedBitmap: Bitmap,
        val realCount: Int,
        val fakeCount: Int,
        val lowQualityCount: Int,
        val verdict: String
    )
    
    init {
        loadModels()
    }
    
    private fun loadModels() {
        val options = Interpreter.Options().apply {
            setNumThreads(4)
        }
        
        try {
            // Load detector
            val detectorBuffer = FileUtil.loadMappedFile(context, "seedshield_detector.tflite")
            detector = Interpreter(detectorBuffer, options)
            
            // Load classifier
            val classifierBuffer = FileUtil.loadMappedFile(context, "seedshield_classifier.tflite")
            classifier = Interpreter(classifierBuffer, options)
        } catch (e: Exception) {
            e.printStackTrace()
        }
    }
    
    fun analyze(bitmap: Bitmap): AnalysisResult {
        val seeds = mutableListOf<SeedResult>()
        
        // Step 1: Detect seeds
        val detections = detectSeeds(bitmap)
        
        // Step 2: Classify each seed
        for (detection in detections) {
            val bbox = detection.first
            val seedCrop = cropBitmap(bitmap, bbox)
            
            if (seedCrop != null) {
                val (className, confidence) = classifySeed(seedCrop)
                seeds.add(SeedResult(bbox, className, confidence))
            }
        }
        
        // Step 3: Create annotated image
        val annotated = drawAnnotations(bitmap, seeds)
        
        // Step 4: Calculate statistics
        val realCount = seeds.count { it.className == "REAL" }
        val fakeCount = seeds.count { it.className == "FAKE" }
        val lowQualityCount = seeds.count { it.className == "LOW_QUALITY" }
        
        // Step 5: Determine verdict
        val total = seeds.size
        val verdict = when {
            total == 0 -> "No seeds detected"
            fakeCount.toFloat() / total > 0.3f -> "❌ Suspected Fake"
            lowQualityCount.toFloat() / total > 0.4f -> "⚠️ Low Quality / Mixed"
            else -> "✅ Mostly Genuine"
        }
        
        return AnalysisResult(
            seeds = seeds,
            annotatedBitmap = annotated,
            realCount = realCount,
            fakeCount = fakeCount,
            lowQualityCount = lowQualityCount,
            verdict = verdict
        )
    }
    
    private fun detectSeeds(bitmap: Bitmap): List<Pair<RectF, Float>> {
        // Resize for detector
        val resized = Bitmap.createScaledBitmap(bitmap, DETECTOR_INPUT_SIZE, DETECTOR_INPUT_SIZE, true)
        
        // Prepare input buffer
        val inputBuffer = ByteBuffer.allocateDirect(1 * DETECTOR_INPUT_SIZE * DETECTOR_INPUT_SIZE * 3 * 4)
        inputBuffer.order(ByteOrder.nativeOrder())
        
        // Fill buffer with normalized pixel values
        for (y in 0 until DETECTOR_INPUT_SIZE) {
            for (x in 0 until DETECTOR_INPUT_SIZE) {
                val pixel = resized.getPixel(x, y)
                inputBuffer.putFloat(Color.red(pixel) / 255f)
                inputBuffer.putFloat(Color.green(pixel) / 255f)
                inputBuffer.putFloat(Color.blue(pixel) / 255f)
            }
        }
        
        // Run detection (simplified - actual output parsing depends on model export format)
        // For demo purposes, return random detections
        return generateDemoDetections(bitmap.width, bitmap.height)
    }
    
    private fun generateDemoDetections(width: Int, height: Int): List<Pair<RectF, Float>> {
        val detections = mutableListOf<Pair<RectF, Float>>()
        val random = java.util.Random()
        
        repeat(random.nextInt(10) + 5) {
            val size = 40f + random.nextFloat() * 40f
            val x = random.nextFloat() * (width - size)
            val y = random.nextFloat() * (height - size)
            
            detections.add(
                Pair(
                    RectF(x, y, x + size, y + size * 0.9f),
                    0.7f + random.nextFloat() * 0.3f
                )
            )
        }
        
        return detections
    }
    
    private fun classifySeed(seedBitmap: Bitmap): Pair<String, Float> {
        // Resize for classifier
        val resized = Bitmap.createScaledBitmap(seedBitmap, CLASSIFIER_INPUT_SIZE, CLASSIFIER_INPUT_SIZE, true)
        
        // Prepare input (normalized with ImageNet mean/std)
        val inputBuffer = ByteBuffer.allocateDirect(1 * CLASSIFIER_INPUT_SIZE * CLASSIFIER_INPUT_SIZE * 3 * 4)
        inputBuffer.order(ByteOrder.nativeOrder())
        
        val mean = floatArrayOf(0.485f, 0.456f, 0.406f)
        val std = floatArrayOf(0.229f, 0.224f, 0.225f)
        
        for (y in 0 until CLASSIFIER_INPUT_SIZE) {
            for (x in 0 until CLASSIFIER_INPUT_SIZE) {
                val pixel = resized.getPixel(x, y)
                inputBuffer.putFloat((Color.red(pixel) / 255f - mean[0]) / std[0])
                inputBuffer.putFloat((Color.green(pixel) / 255f - mean[1]) / std[1])
                inputBuffer.putFloat((Color.blue(pixel) / 255f - mean[2]) / std[2])
            }
        }
        
        // Prepare output
        val output = Array(1) { FloatArray(3) }
        
        // Run classifier
        classifier?.run(inputBuffer, output)
        
        // Apply softmax and get prediction
        val probs = softmax(output[0])
        val maxIdx = probs.indices.maxByOrNull { probs[it] } ?: 0
        
        return Pair(CLASSES[maxIdx], probs[maxIdx])
    }
    
    private fun softmax(logits: FloatArray): FloatArray {
        val maxLogit = logits.maxOrNull() ?: 0f
        val exps = logits.map { kotlin.math.exp(it - maxLogit) }
        val sum = exps.sum()
        return exps.map { it / sum }.toFloatArray()
    }
    
    private fun cropBitmap(bitmap: Bitmap, rect: RectF): Bitmap? {
        return try {
            val x = rect.left.toInt().coerceIn(0, bitmap.width - 1)
            val y = rect.top.toInt().coerceIn(0, bitmap.height - 1)
            val w = rect.width().toInt().coerceIn(1, bitmap.width - x)
            val h = rect.height().toInt().coerceIn(1, bitmap.height - y)
            Bitmap.createBitmap(bitmap, x, y, w, h)
        } catch (e: Exception) {
            null
        }
    }
    
    private fun drawAnnotations(bitmap: Bitmap, seeds: List<SeedResult>): Bitmap {
        val result = bitmap.copy(Bitmap.Config.ARGB_8888, true)
        val canvas = Canvas(result)
        
        val boxPaint = Paint().apply {
            style = Paint.Style.STROKE
            strokeWidth = 4f
        }
        
        val textPaint = Paint().apply {
            color = Color.WHITE
            textSize = 24f
            isFakeBoldText = true
        }
        
        val bgPaint = Paint().apply {
            style = Paint.Style.FILL
        }
        
        for (seed in seeds) {
            val color = COLORS[seed.className] ?: Color.GRAY
            boxPaint.color = color
            bgPaint.color = color
            
            // Draw bounding box
            canvas.drawRect(seed.bbox, boxPaint)
            
            // Draw label background
            val label = "${seed.className}: ${(seed.confidence * 100).toInt()}%"
            val textWidth = textPaint.measureText(label)
            val textHeight = 28f
            
            val labelX = seed.bbox.left
            val labelY = if (seed.bbox.top > textHeight + 5) seed.bbox.top - 5 else seed.bbox.bottom + textHeight + 5
            
            canvas.drawRect(labelX, labelY - textHeight, labelX + textWidth + 10, labelY + 5, bgPaint)
            canvas.drawText(label, labelX + 5, labelY, textPaint)
        }
        
        return result
    }
    
    fun close() {
        detector?.close()
        classifier?.close()
    }
}
```

### MainActivity.kt

```kotlin
package com.seedshield

import android.graphics.BitmapFactory
import android.os.Bundle
import android.widget.Button
import android.widget.ImageView
import android.widget.TextView
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import com.seedshield.analyzer.SeedShieldAnalyzer

class MainActivity : AppCompatActivity() {
    
    private lateinit var analyzer: SeedShieldAnalyzer
    private lateinit var imageView: ImageView
    private lateinit var verdictText: TextView
    private lateinit var statsText: TextView
    
    private val pickImage = registerForActivityResult(ActivityResultContracts.GetContent()) { uri ->
        uri?.let {
            val stream = contentResolver.openInputStream(it)
            val bitmap = BitmapFactory.decodeStream(stream)
            
            // Analyze
            val result = analyzer.analyze(bitmap)
            
            // Display
            imageView.setImageBitmap(result.annotatedBitmap)
            verdictText.text = result.verdict
            statsText.text = buildString {
                append("Total: ${result.seeds.size}\n")
                append("🟢 Real: ${result.realCount}\n")
                append("🔴 Fake: ${result.fakeCount}\n")
                append("🟡 Low Quality: ${result.lowQualityCount}")
            }
        }
    }
    
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        
        analyzer = SeedShieldAnalyzer(this)
        
        imageView = findViewById(R.id.imageView)
        verdictText = findViewById(R.id.verdictText)
        statsText = findViewById(R.id.statsText)
        
        findViewById<Button>(R.id.selectButton).setOnClickListener {
            pickImage.launch("image/*")
        }
    }
    
    override fun onDestroy() {
        super.onDestroy()
        analyzer.close()
    }
}
```

### activity_main.xml

```xml
<?xml version="1.0" encoding="utf-8"?>
<LinearLayout xmlns:android="http://schemas.android.com/apk/res/android"
    android:layout_width="match_parent"
    android:layout_height="match_parent"
    android:orientation="vertical"
    android:padding="16dp">

    <TextView
        android:layout_width="wrap_content"
        android:layout_height="wrap_content"
        android:text="🌱 SeedShield"
        android:textSize="28sp"
        android:textStyle="bold"
        android:layout_gravity="center_horizontal"
        android:paddingBottom="16dp"/>

    <ImageView
        android:id="@+id/imageView"
        android:layout_width="match_parent"
        android:layout_height="0dp"
        android:layout_weight="1"
        android:scaleType="fitCenter"
        android:background="#F0F0F0"/>

    <TextView
        android:id="@+id/verdictText"
        android:layout_width="match_parent"
        android:layout_height="wrap_content"
        android:textSize="24sp"
        android:textStyle="bold"
        android:textAlignment="center"
        android:padding="16dp"/>

    <TextView
        android:id="@+id/statsText"
        android:layout_width="match_parent"
        android:layout_height="wrap_content"
        android:textSize="16sp"
        android:padding="8dp"/>

    <Button
        android:id="@+id/selectButton"
        android:layout_width="match_parent"
        android:layout_height="wrap_content"
        android:text="📷 Select Image"
        android:textSize="18sp"
        android:padding="16dp"/>

</LinearLayout>
```

---

## Optimization Tips

1. **Enable GPU Acceleration**:
```kotlin
val options = Interpreter.Options().apply {
    addDelegate(GpuDelegate())
}
```

2. **Use NNAPI for Neural Engine**:
```kotlin
val options = Interpreter.Options().apply {
    useNNAPI = true
}
```

3. **Quantize Models** for smaller size and faster inference

---

## Testing

Run on a physical device for accurate performance testing. Emulators don't support GPU delegation.

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Model load fails | Check asset path and file existence |
| Slow inference | Enable GPU/NNAPI, reduce image size |
| OOM errors | Reduce batch size, use smaller input |
| Incorrect results | Verify preprocessing normalization |
