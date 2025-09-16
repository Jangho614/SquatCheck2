// PoseMath.kt
package com.google.mediapipe.examples.poselandmarker

import kotlin.math.abs
import kotlin.math.atan2
import kotlin.math.PI

data class Pt(val x: Double, val y: Double)

fun calculateAngle(a: Pt, b: Pt, c: Pt): Double {
    val radians = atan2(c.y - b.y, c.x - b.x) - atan2(a.y - b.y, a.x - b.x)
    var angle = abs(radians * 180.0 / PI)
    if (angle > 180.0) angle = 360.0 - angle
    return angle
}

data class PoseAngles(
    val Rknee_angle: Double,
    val Lknee_angle: Double,
    val Rhip_angle: Double,
    val Lhip_angle: Double
)
