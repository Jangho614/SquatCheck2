/*
 * Copyright 2023 The TensorFlow Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package com.google.mediapipe.examples.poselandmarker

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.util.AttributeSet
import android.view.View
import androidx.core.content.ContextCompat
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.poselandmarker.PoseLandmarker
import com.google.mediapipe.tasks.vision.poselandmarker.PoseLandmarkerResult
import kotlin.math.max
import kotlin.math.min

var onAnglesComputed: ((PoseAngles) -> Unit)? = null

class OverlayView(context: Context?, attrs: AttributeSet?) :
    View(context, attrs) {

    private var results: PoseLandmarkerResult? = null
    private var pointPaint = Paint()
    private var linePaint = Paint()

    private var scaleFactor: Float = 1f
    private var imageWidth: Int = 1
    private var imageHeight: Int = 1

    init {
        initPaints()
    }

    fun clear() {
        results = null
        pointPaint.reset()
        linePaint.reset()
        invalidate()
        initPaints()
    }

    private fun initPaints() {
        linePaint.color =
            ContextCompat.getColor(context!!, R.color.mp_color_primary)
        linePaint.strokeWidth = LANDMARK_STROKE_WIDTH
        linePaint.style = Paint.Style.STROKE

        pointPaint.color = Color.YELLOW
        pointPaint.strokeWidth = LANDMARK_STROKE_WIDTH
        pointPaint.style = Paint.Style.FILL
    }

    override fun draw(canvas: Canvas) {
        super.draw(canvas)
        results?.let { poseLandmarkerResult ->
            for(landmark in poseLandmarkerResult.landmarks()) {
                for(normalizedLandmark in landmark) {
                    canvas.drawPoint(
                        normalizedLandmark.x() * imageWidth * scaleFactor,
                        normalizedLandmark.y() * imageHeight * scaleFactor,
                        pointPaint
                    )
                }

                PoseLandmarker.POSE_LANDMARKS.forEach {
                    canvas.drawLine(
                        poseLandmarkerResult.landmarks().get(0).get(it!!.start()).x() * imageWidth * scaleFactor,
                        poseLandmarkerResult.landmarks().get(0).get(it.start()).y() * imageHeight * scaleFactor,
                        poseLandmarkerResult.landmarks().get(0).get(it.end()).x() * imageWidth * scaleFactor,
                        poseLandmarkerResult.landmarks().get(0).get(it.end()).y() * imageHeight * scaleFactor,
                        linePaint)
                }
            }
        }
    }

    fun setResults(
        poseLandmarkerResults: PoseLandmarkerResult,
        imageHeight: Int,
        imageWidth: Int,
        runningMode: RunningMode = RunningMode.IMAGE
    ) {
        results = poseLandmarkerResults

        this.imageHeight = imageHeight
        this.imageWidth = imageWidth

        scaleFactor = when (runningMode) {
            RunningMode.IMAGE,
            RunningMode.VIDEO -> {
                min(width * 1f / imageWidth, height * 1f / imageHeight)
            }
            RunningMode.LIVE_STREAM -> {
                // PreviewView is in FILL_START mode. So we need to scale up the
                // landmarks to match with the size that the captured images will be
                // displayed.
                max(width * 1f / imageWidth, height * 1f / imageHeight)
            }
        }
        val hasAny = results?.landmarks()?.isNotEmpty() == true
        if (!hasAny) {
            invalidate()
            return
        }
        // OverlayView.kt -> setResults(...) 안
        val joints = extractJointsAsPixels(results!!)
        val angles = computeAngles(joints)
        onAnglesComputed?.invoke(angles)   // 외부로 각도 전달 (필요 없으면 이 줄은 생략 가능)

        invalidate()
    }

    companion object {
        private const val LANDMARK_STROKE_WIDTH = 12F
    }

    // 관절 좌표 추출 (이미지 픽셀 좌표계로 꺼냄)
    private fun extractJointsAsPixels(result: PoseLandmarkerResult): Map<String, Pt> {
        val person = result.landmarks()[0]  // 첫 번째 사람 기준
        fun lm(i: Int) = Pt(
            x = person[i].x().toDouble() * imageWidth,
            y = person[i].y().toDouble() * imageHeight
        )

        return mapOf(
            "Rshoulder" to lm(12),
            "Lshoulder" to lm(11),
            "Rhip"      to lm(24),
            "Lhip"      to lm(23),
            "Rknee"     to lm(26),
            "Lknee"     to lm(25),
            "Rankle"    to lm(28),
            "Lankle"    to lm(27)
        )
    }

    private fun computeAngles(j: Map<String, Pt>): PoseAngles {
        val Rknee_angle = 180 - calculateAngle(j["Rhip"]!!, j["Rknee"]!!, j["Rankle"]!!)
        val Lknee_angle = 180 - calculateAngle(j["Lhip"]!!, j["Lknee"]!!, j["Lankle"]!!)
        val Rhip_angle  = 180 - calculateAngle(j["Rshoulder"]!!, j["Rhip"]!!, j["Rknee"]!!)
        val Lhip_angle  = 180 - calculateAngle(j["Lshoulder"]!!, j["Lhip"]!!, j["Lknee"]!!)
        return PoseAngles(Rknee_angle, Lknee_angle, Rhip_angle, Lhip_angle)
    }

}