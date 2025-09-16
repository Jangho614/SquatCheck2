package com.google.mediapipe.examples.poselandmarker.fragment

import android.annotation.SuppressLint
import android.content.res.Configuration
import android.os.Bundle
import android.util.Log
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.Toast
import androidx.camera.core.Preview
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Camera
import androidx.camera.core.AspectRatio
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.content.ContextCompat
import androidx.fragment.app.Fragment
import androidx.fragment.app.activityViewModels
import androidx.navigation.Navigation
import com.google.mediapipe.examples.poselandmarker.PoseLandmarkerHelper
import com.google.mediapipe.examples.poselandmarker.MainViewModel
import com.google.mediapipe.examples.poselandmarker.R
import com.google.mediapipe.examples.poselandmarker.databinding.FragmentCameraBinding
import com.google.mediapipe.tasks.vision.core.RunningMode
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import kotlin.math.abs
import kotlin.math.atan2
import kotlin.math.PI
import java.util.Locale

// ⬇️ 추가: 분류기
import com.google.mediapipe.examples.poselandmarker.classifier.SquatClassifier

class CameraFragment : Fragment(), PoseLandmarkerHelper.LandmarkerListener {

    companion object {
        private const val TAG = "Pose Landmarker"
        private const val MIN_INFER_INTERVAL_MS = 100L  // 추론 최소 간격(너무 과도한 호출 방지)
    }

    private var _fragmentCameraBinding: FragmentCameraBinding? = null
    private val fragmentCameraBinding get() = _fragmentCameraBinding!!

    private lateinit var poseLandmarkerHelper: PoseLandmarkerHelper
    private val viewModel: MainViewModel by activityViewModels()
    private var preview: Preview? = null
    private var imageAnalyzer: ImageAnalysis? = null
    private var camera: Camera? = null
    private var cameraProvider: ProcessCameraProvider? = null
    private var cameraFacing = CameraSelector.LENS_FACING_BACK

    /** ML/분류 등 블로킹 작업용 실행기 */
    private lateinit var backgroundExecutor: ExecutorService

    /** ⬇️ 추가: TFLite 분류기 */
    private var classifier: SquatClassifier? = null
    private var lastInferTime = 0L

    override fun onResume() {
        super.onResume()
        if (!PermissionsFragment.hasPermissions(requireContext())) {
            Navigation.findNavController(requireActivity(), R.id.fragment_container)
                .navigate(R.id.action_camera_to_permissions)
        }

        // 돌아왔을 때 헬퍼가 닫혀 있으면 다시 준비
        backgroundExecutor.execute {
            if (this::poseLandmarkerHelper.isInitialized) {
                if (poseLandmarkerHelper.isClose()) {
                    poseLandmarkerHelper.setupPoseLandmarker()
                }
            }
        }
    }

    override fun onPause() {
        super.onPause()
        if (this::poseLandmarkerHelper.isInitialized) {
            // 현재 값 저장(필요시)
            viewModel.setMinPoseDetectionConfidence(poseLandmarkerHelper.minPoseDetectionConfidence)
            viewModel.setMinPoseTrackingConfidence(poseLandmarkerHelper.minPoseTrackingConfidence)
            viewModel.setMinPosePresenceConfidence(poseLandmarkerHelper.minPosePresenceConfidence)
            viewModel.setDelegate(poseLandmarkerHelper.currentDelegate)

            backgroundExecutor.execute { poseLandmarkerHelper.clearPoseLandmarker() }
        }
    }

    override fun onDestroyView() {
        _fragmentCameraBinding = null
        super.onDestroyView()

        // 분류기 자원 해제
        classifier?.close()
        classifier = null

        // 백그라운드 실행기 종료
        backgroundExecutor.shutdown()
        backgroundExecutor.awaitTermination(Long.MAX_VALUE, TimeUnit.NANOSECONDS)
    }

    override fun onCreateView(
        inflater: LayoutInflater,
        container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View {
        _fragmentCameraBinding = FragmentCameraBinding.inflate(inflater, container, false)
        return fragmentCameraBinding.root
    }

    @SuppressLint("MissingPermission")
    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)

        // 백그라운드 실행기
        backgroundExecutor = Executors.newSingleThreadExecutor()

        // View가 배치된 다음 카메라 셋업
        fragmentCameraBinding.viewFinder.post { setUpCamera() }

        // PoseLandmarkerHelper 생성 + 모델 준비
        backgroundExecutor.execute {
            poseLandmarkerHelper = PoseLandmarkerHelper(
                context = requireContext(),
                runningMode = RunningMode.LIVE_STREAM,
                minPoseDetectionConfidence = 0.5f,
                minPoseTrackingConfidence = 0.5f,
                minPosePresenceConfidence = 0.5f,
                currentDelegate = PoseLandmarkerHelper.DELEGATE_CPU,
                poseLandmarkerHelperListener = this
            )
            try {
                // Lite 모델 인덱스 가정(필요 시 상수 사용)
                poseLandmarkerHelper.currentModel = 0
                poseLandmarkerHelper.clearPoseLandmarker()
                poseLandmarkerHelper.setupPoseLandmarker()
            } catch (e: Exception) {
                Log.w(TAG, "모델 설정 중 예외 발생: ${e.message}")
            }

            // ⬇️ TFLite 분류기 준비(assets/model.tflite 가정)
            try {
                classifier = SquatClassifier(requireContext())
                Log.i(TAG, "SquatClassifier initialized.")
            } catch (e: Exception) {
                Log.e(TAG, "분류기 초기화 실패: ${e.message}")
            }
        }
    }

    // CameraX 준비
    private fun setUpCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(requireContext())
        cameraProviderFuture.addListener(
            {
                cameraProvider = cameraProviderFuture.get()
                bindCameraUseCases()
            },
            ContextCompat.getMainExecutor(requireContext())
        )
    }

    // Preview + Analysis 바인딩
    @SuppressLint("UnsafeOptInUsageError")
    private fun bindCameraUseCases() {
        val cameraProvider = cameraProvider ?: throw IllegalStateException("Camera init failed.")

        val cameraSelector = CameraSelector.Builder()
            .requireLensFacing(cameraFacing)
            .build()

        preview = Preview.Builder()
            .setTargetAspectRatio(AspectRatio.RATIO_4_3)
            .setTargetRotation(fragmentCameraBinding.viewFinder.display.rotation)
            .build()

        imageAnalyzer = ImageAnalysis.Builder()
            .setTargetAspectRatio(AspectRatio.RATIO_4_3)
            .setTargetRotation(fragmentCameraBinding.viewFinder.display.rotation)
            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
            .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
            .build()
            .also {
                it.setAnalyzer(backgroundExecutor) { image -> detectPose(image) }
            }

        cameraProvider.unbindAll()

        try {
            camera = cameraProvider.bindToLifecycle(
                this, cameraSelector, preview, imageAnalyzer
            )
            preview?.setSurfaceProvider(fragmentCameraBinding.viewFinder.surfaceProvider)
        } catch (exc: Exception) {
            Log.e(TAG, "Use case binding failed", exc)
        }
    }

    private fun detectPose(imageProxy: ImageProxy) {
        if (this::poseLandmarkerHelper.isInitialized) {
            poseLandmarkerHelper.detectLiveStream(
                imageProxy = imageProxy,
                isFrontCamera = cameraFacing == CameraSelector.LENS_FACING_FRONT
            )
        } else {
            imageProxy.close()
        }
    }

    override fun onConfigurationChanged(newConfig: Configuration) {
        super.onConfigurationChanged(newConfig)
        imageAnalyzer?.targetRotation = fragmentCameraBinding.viewFinder.display.rotation
    }

    // 결과 수신: 오버레이 갱신 + 좌표/각도 → 분류기 호출
    override fun onResults(resultBundle: PoseLandmarkerHelper.ResultBundle) {
        activity?.runOnUiThread {
            if (_fragmentCameraBinding == null) return@runOnUiThread

            val resultsList = resultBundle.results
            if (resultsList.isEmpty()) {
                Log.d(TAG, "No landmarks in this frame")
                fragmentCameraBinding.overlay.clear()
                fragmentCameraBinding.overlay.invalidate()
                return@runOnUiThread
            }

            val poseResult = resultsList.first()

            fragmentCameraBinding.overlay.setResults(
                poseResult,
                resultBundle.inputImageHeight,
                resultBundle.inputImageWidth,
                RunningMode.LIVE_STREAM
            )

            // 좌표 + 각도 추출
            val log = extractPoseLog(
                poseResult,
                resultBundle.inputImageWidth,
                resultBundle.inputImageHeight
            )

            // (선택) 디버그: 원본 feature 로깅
            if (log != null) {
                Log.d(
                    TAG, String.format(
                        Locale.US,
                        "Rshoulder_x=%.1f, Rshoulder_y=%.1f, Lshoulder_x=%.1f, Lshoulder_y=%.1f, " +
                                "Rhip_x=%.1f, Rhip_y=%.1f, Lhip_x=%.1f, Lhip_y=%.1f, " +
                                "Rknee_x=%.1f, Rknee_y=%.1f, Lknee_x=%.1f, Lknee_y=%.1f, " +
                                "Rankle_x=%.1f, Rankle_y=%.1f, Lankle_x=%.1f, Lankle_y=%.1f, " +
                                "Rknee_angle=%.1f, Lknee_angle=%.1f, Rhip_angle=%.1f, Lhip_angle=%.1f",
                        log.Rshoulder_x, log.Rshoulder_y, log.Lshoulder_x, log.Lshoulder_y,
                        log.Rhip_x, log.Rhip_y, log.Lhip_x, log.Lhip_y,
                        log.Rknee_x, log.Rknee_y, log.Lknee_x, log.Lknee_y,
                        log.Rankle_x, log.Rankle_y, log.Lankle_x, log.Lankle_y,
                        log.Rknee_angle, log.Lknee_angle, log.Rhip_angle, log.Lhip_angle
                    )
                )
            }

            // 분류 호출 (throttle)
            val now = System.currentTimeMillis()
            if (log != null && (now - lastInferTime) >= MIN_INFER_INTERVAL_MS) {
                lastInferTime = now

                // ⬇️ 분류 입력 vector 생성
                val features = floatArrayOf(
                    log.Rshoulder_x.toFloat(), log.Rshoulder_y.toFloat(),
                    log.Lshoulder_x.toFloat(), log.Lshoulder_y.toFloat(),
                    log.Rhip_x.toFloat(), log.Rhip_y.toFloat(),
                    log.Lhip_x.toFloat(), log.Lhip_y.toFloat(),
                    log.Rknee_x.toFloat(), log.Rknee_y.toFloat(),
                    log.Lknee_x.toFloat(), log.Lknee_y.toFloat(),
                    log.Rankle_x.toFloat(), log.Rankle_y.toFloat(),
                    log.Lankle_x.toFloat(), log.Lankle_y.toFloat(),
                    log.Rknee_angle.toFloat(), log.Lknee_angle.toFloat(),
                    log.Rhip_angle.toFloat(), log.Lhip_angle.toFloat()
                )

                // ⬇️ 백그라운드에서 추론 실행
                backgroundExecutor.execute {
                    val pred = try {
                        classifier?.predict(features) ?: -1
                    } catch (e: Exception) {
                        Log.e(TAG, "분류 실패: ${e.message}")
                        -1
                    }
                    Log.d("SquatClassifier", "Result = $pred") // 0/1/2 출력
                }
            }

            fragmentCameraBinding.overlay.invalidate()
        }
    }

    override fun onError(error: String, errorCode: Int) {
        activity?.runOnUiThread {
            Toast.makeText(requireContext(), error, Toast.LENGTH_SHORT).show()
        }
        // GPU 오류 시 CPU로 자동 전환
        if (errorCode == PoseLandmarkerHelper.GPU_ERROR && this::poseLandmarkerHelper.isInitialized) {
            backgroundExecutor.execute {
                try {
                    poseLandmarkerHelper.currentDelegate = PoseLandmarkerHelper.DELEGATE_CPU
                    poseLandmarkerHelper.clearPoseLandmarker()
                    poseLandmarkerHelper.setupPoseLandmarker()
                    Log.i(TAG, "GPU 오류로 CPU delegate로 자동 전환")
                } catch (e: Exception) {
                    Log.e(TAG, "CPU 전환 실패: ${e.message}")
                }
            }
        }
    }

    // --------------------------
    // ⬇️ 각도/로깅 헬퍼
    // --------------------------
    data class Pt(val x: Double, val y: Double)
    data class PoseAngles(
        val Rknee_angle: Double,
        val Lknee_angle: Double,
        val Rhip_angle: Double,
        val Lhip_angle: Double
    )

    private fun calculateAngle(a: Pt, b: Pt, c: Pt): Double {
        val radians = atan2(c.y - b.y, c.x - b.x) - atan2(a.y - b.y, a.x - b.x)
        var angle = abs(radians * 180.0 / PI)
        if (angle > 180.0) angle = 360.0 - angle
        return angle
    }

    private fun computeAnglesFrom(
        result: com.google.mediapipe.tasks.vision.poselandmarker.PoseLandmarkerResult,
        imageWidth: Int,
        imageHeight: Int
    ): PoseAngles {
        val lm = result.landmarks().firstOrNull()
            ?: return PoseAngles(0.0, 0.0, 0.0, 0.0)

        fun p(i: Int) = Pt(
            x = lm[i].x().toDouble() * imageWidth,
            y = lm[i].y().toDouble() * imageHeight
        )

        val Rshoulder = p(12)
        val Lshoulder = p(11)
        val Rhip = p(24)
        val Lhip = p(23)
        val Rknee = p(26)
        val Lknee = p(25)
        val Rankle = p(28)
        val Lankle = p(27)

        val RkneeAngle = 180 - calculateAngle(Rhip, Rknee, Rankle)
        val LkneeAngle = 180 - calculateAngle(Lhip, Lknee, Lankle)
        val RhipAngle  = 180 - calculateAngle(Rshoulder, Rhip, Rknee)
        val LhipAngle  = 180 - calculateAngle(Lshoulder, Lhip, Lknee)

        return PoseAngles(RkneeAngle, LkneeAngle, RhipAngle, LhipAngle)
    }

    // 16개 좌표 + 4개 각도 한 번에 담는 구조
    data class PoseLog(
        val Rshoulder_x: Double, val Rshoulder_y: Double,
        val Lshoulder_x: Double, val Lshoulder_y: Double,
        val Rhip_x: Double, val Rhip_y: Double,
        val Lhip_x: Double, val Lhip_y: Double,
        val Rknee_x: Double, val Rknee_y: Double,
        val Lknee_x: Double, val Lknee_y: Double,
        val Rankle_x: Double, val Rankle_y: Double,
        val Lankle_x: Double, val Lankle_y: Double,
        val Rknee_angle: Double, val Lknee_angle: Double,
        val Rhip_angle: Double, val Lhip_angle: Double
    )

    // 결과에서 좌표/각도 모두 추출 (없거나 부족하면 null)
    private fun extractPoseLog(
        result: com.google.mediapipe.tasks.vision.poselandmarker.PoseLandmarkerResult,
        imageWidth: Int,
        imageHeight: Int
    ): PoseLog? {
        val lm = result.landmarks().firstOrNull() ?: return null
        if (lm.size <= 28) return null  // 인덱스 안전 가드

        fun p(i: Int) = Pt(
            x = lm[i].x().toDouble() * imageWidth,
            y = lm[i].y().toDouble() * imageHeight
        )

        val Rshoulder = p(12); val Lshoulder = p(11)
        val Rhip      = p(24); val Lhip      = p(23)
        val Rknee     = p(26); val Lknee     = p(25)
        val Rankle    = p(28); val Lankle    = p(27)

        val RkneeAngle = 180 - calculateAngle(Rhip, Rknee, Rankle)
        val LkneeAngle = 180 - calculateAngle(Lhip, Lknee, Lankle)
        val RhipAngle  = 180 - calculateAngle(Rshoulder, Rhip, Rknee)
        val LhipAngle  = 180 - calculateAngle(Lshoulder, Lhip, Lknee)

        return PoseLog(
            Rshoulder.x, Rshoulder.y,
            Lshoulder.x, Lshoulder.y,
            Rhip.x, Rhip.y,
            Lhip.x, Lhip.y,
            Rknee.x, Rknee.y,
            Lknee.x, Lknee.y,
            Rankle.x, Rankle.y,
            Lankle.x, Lankle.y,
            RkneeAngle, LkneeAngle, RhipAngle, LhipAngle
        )
    }
}
