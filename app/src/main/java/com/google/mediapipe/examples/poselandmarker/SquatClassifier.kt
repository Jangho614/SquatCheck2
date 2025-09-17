package com.google.mediapipe.examples.poselandmarker.classifier

import android.content.Context
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

class SquatClassifier(context: Context, modelAssetPath: String = "squat_model_with_scaler.tflite") {
    private val interpreter: Interpreter

    init {
        val assetFileDescriptor = context.assets.openFd(modelAssetPath)
        val inputStream = FileInputStream(assetFileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        val startOffset = assetFileDescriptor.startOffset
        val declaredLength = assetFileDescriptor.declaredLength
        val model: MappedByteBuffer =
            fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
        interpreter = Interpreter(model)
    }

    fun predict(features: FloatArray): Int {
        require(features.size == 20) { "입력은 반드시 20개 float 값이어야 합니다." }

        val input = arrayOf(features)          // [1,20]
        val output = Array(1) { FloatArray(3) } // [1,3] → 클래스 확률

        interpreter.run(input, output)

        return output[0].indices.maxByOrNull { output[0][it] } ?: -1
    }

    fun close() {
        interpreter.close()
    }
}
