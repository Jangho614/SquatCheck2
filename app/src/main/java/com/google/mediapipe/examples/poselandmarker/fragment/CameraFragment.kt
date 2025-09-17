package com.google.mediapipe.examples.poselandmarker.fragment

import android.annotation.SuppressLint
import android.content.res.Configuration
import android.graphics.Color
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

// ⬇️ 분류기
import com.google.mediapipe.examples.poselandmarker.classifier.SquatClassifier
import com.google.mediapipe.tasks.vision.poselandmarker.PoseLandmarkerResult

class CameraFragment : Fragment(), PoseLandmarkerHelper.LandmarkerListener {

    companion object {
        private const val TAG = "Pose Landmarker"
        private const val MIN_INFER_INTERVAL_MS = 100L
    }

    private var _fragmentCameraBinding: FragmentCameraBinding? = null
    private val fragmentCameraBinding get() = _fragmentCameraBinding!!

    private lateinit var poseLandmarkerHelper: PoseLandmarkerHelper
    private val viewModel: MainViewModel by activityViewModels()
    private var preview: Preview? = null
    private var imageAnalyzer: ImageAnalysis? = null
    private var camera: Camera? = null
    private var cameraProvider: ProcessCameraProvider? = null

    // ✅ 기본 카메라 전면
    private var cameraFacing = CameraSelector.LENS_FACING_FRONT

    private lateinit var backgroundExecutor: ExecutorService

    private var classifier: SquatClassifier? = null
    private var lastInferTime = 0L

    // =====================
    // 스쿼트 카운팅 FSM 상태
    // =====================
    private var squatCount = 0
    private var inDownPhase = false          // DOWN(앉음) 구간 진입 여부
    private var stableState = -1             // 최근 확정된 상태(0/1/2)
    private var sameStateCount = 0           // 같은 상태가 연속 등장한 프레임 수
    private val REQUIRED_STABLE_FRAMES = 3   // 상태 확정을 위한 최소 연속 프레임 수

    private var wrongSquatCount = 0
    private var correctSquatCount = 0

    private var squatGoal = 100

    private val STATE_STAND = 0
    private val STATE_GOOD  = 1
    private val STATE_BAD   = 2
    private fun isSquatState(s: Int) = (s == STATE_GOOD || s == STATE_BAD)

    override fun onResume() {
        super.onResume()
        if (!PermissionsFragment.hasPermissions(requireContext())) {
            Navigation.findNavController(requireActivity(), R.id.fragment_container)
                .navigate(R.id.action_camera_to_permissions)
        }

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

        classifier?.close()
        classifier = null

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

    // 전면/후면 지원 여부 확인
    private fun hasLens(provider: ProcessCameraProvider, lensFacing: Int): Boolean {
        val selector = CameraSelector.Builder().requireLensFacing(lensFacing).build()
        return try {
            provider.hasCamera(selector)
        } catch (_: Exception) {
            false
        }
    }

    @SuppressLint("MissingPermission")
    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)

        backgroundExecutor = Executors.newSingleThreadExecutor()
        fragmentCameraBinding.viewFinder.post { setUpCamera() }

        // 스쿼트 카운트 초기 표시
        fragmentCameraBinding.txtSquatCount.text = "0"

        // 🔽 카메라 전환 버튼
        fragmentCameraBinding.ChangeCamera.setOnClickListener {
            val provider = cameraProvider
            if (provider == null) {
                Toast.makeText(requireContext(), "카메라 초기화 중입니다.", Toast.LENGTH_SHORT).show()
                return@setOnClickListener
            }

            val nextFacing =
                if (cameraFacing == CameraSelector.LENS_FACING_FRONT)
                    CameraSelector.LENS_FACING_BACK
                else
                    CameraSelector.LENS_FACING_FRONT

            if (!hasLens(provider, nextFacing)) {
                val msg = if (nextFacing == CameraSelector.LENS_FACING_FRONT)
                    "전면 카메라를 찾을 수 없습니다."
                else
                    "후면 카메라를 찾을 수 없습니다."
                Toast.makeText(requireContext(), msg, Toast.LENGTH_SHORT).show()
                return@setOnClickListener
            }

            cameraFacing = nextFacing
            try {
                provider.unbindAll()
                bindCameraUseCases()
                val label = if (cameraFacing == CameraSelector.LENS_FACING_FRONT) "전면" else "후면"
                Toast.makeText(requireContext(), "카메라 전환: $label", Toast.LENGTH_SHORT).show()
            } catch (e: Exception) {
                Log.e(TAG, "카메라 전환 실패", e)
                Toast.makeText(requireContext(), "카메라 전환에 실패했습니다.", Toast.LENGTH_SHORT).show()
            }
        }

        // ⬇️ 기존 초기화 로직
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
                poseLandmarkerHelper.currentModel = 0
                poseLandmarkerHelper.clearPoseLandmarker()
                poseLandmarkerHelper.setupPoseLandmarker()
            } catch (e: Exception) {
                Log.w(TAG, "모델 설정 중 예외: ${e.message}")
            }

            try {
                classifier = SquatClassifier(requireContext())
                Log.i(TAG, "SquatClassifier initialized.")
            } catch (e: Exception) {
                Log.e(TAG, "분류기 초기화 실패: ${e.message}")
            }
        }
    }

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

    @SuppressLint("UnsafeOptInUsageError")
    private fun bindCameraUseCases() {
        val cameraProvider = cameraProvider ?: throw IllegalStateException("Camera init failed.")

        val cameraSelector = CameraSelector.Builder()
            .requireLensFacing(cameraFacing) // ✅ 현재 렌즈 사용
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
                // ✅ 전면 카메라 여부 전달(표시 미러링은 PreviewView가 처리)
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

    // 결과 수신
    override fun onResults(resultBundle: PoseLandmarkerHelper.ResultBundle) {
        activity?.runOnUiThread {
            if (_fragmentCameraBinding == null) return@runOnUiThread

            val resultsList = resultBundle.results
            if (resultsList.isEmpty()) {
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

            val imgW = resultBundle.inputImageWidth
            val imgH = resultBundle.inputImageHeight
            val log = extractPoseLog(poseResult, imgW, imgH)

            val now = System.currentTimeMillis()
            if (log != null && (now - lastInferTime) >= MIN_INFER_INTERVAL_MS) {
                lastInferTime = now
                val features = toFeaturesForTFLite(log, imgW, imgH)

                // 🔧 예측은 백그라운드에서
                backgroundExecutor.execute {
                    val pred = try {
                        classifier?.predict(features) ?: -1
                    } catch (e: Exception) {
                        Log.e(TAG, "분류 실패: ${e.message}", e)
                        -1
                    }
                    Log.d("SquatClassifier", "Result = $pred") // 0/1/2

                    // ✅ UI 갱신은 전부 메인에서
                    activity?.runOnUiThread {
                        if (!isAdded || _fragmentCameraBinding == null) return@runOnUiThread

                        when (pred) {
                            1 -> {
                                correctSquatCount++
                                fragmentCameraBinding.txtCurrentScore.text = "올바른 자세"
                                fragmentCameraBinding.txtCurrentScore.setTextColor(Color.rgb(70, 150, 250))
                                fragmentCameraBinding.txtAverageScore.setTextColor(Color.rgb(70, 150, 250))
                            }
                            2 -> {
                                wrongSquatCount++
                                fragmentCameraBinding.txtCurrentScore.text = "잘못된 자세"
                                fragmentCameraBinding.txtCurrentScore.setTextColor(Color.rgb(250, 65, 65))
                                fragmentCameraBinding.txtAverageScore.setTextColor(Color.rgb(250, 65, 65))
                            }
                            else -> {
                                fragmentCameraBinding.txtCurrentScore.text = "스쿼트!"
                                fragmentCameraBinding.txtCurrentScore.setTextColor(Color.rgb(70, 70, 70))
                                fragmentCameraBinding.txtAverageScore.setTextColor(Color.rgb(70, 70, 70))
                            }
                        }

                        val total = correctSquatCount + wrongSquatCount
                        val pct = if (total > 0) (correctSquatCount.toFloat() / total) * 100f else 0f
                        fragmentCameraBinding.txtAverageScore.text =
                            String.format(java.util.Locale.getDefault(), "%.1f", pct)
                    }

                    // ⚠️ 이 메서드가 뷰를 만지지 않는 경우에만 여기서 호출
                    // 뷰를 만진다면 위의 runOnUiThread 블록 안으로 옮기세요.
                    handlePrediction(pred)
                }
            }

            fragmentCameraBinding.overlay.invalidate()
        }
    }



    // =====================
    // 스쿼트 카운팅 FSM 로직
    // =====================
    private fun handlePrediction(pred: Int) {
        if (pred !in 0..2) return // 유효하지 않은 결과 무시

        // ----- 상태 안정화(노이즈 억제) -----
        if (pred == stableState) {
            sameStateCount++
        } else {
            stableState = pred
            sameStateCount = 1
        }
        if (sameStateCount < REQUIRED_STABLE_FRAMES) return

        // ----- FSM 전이 -----
        // (1) 아직 DownPhase가 아닌데 1/2 상태가 안정적으로 감지되면 DownPhase 진입
        if (!inDownPhase && isSquatState(stableState)) {
            inDownPhase = true
            return
        }

        // (2) DownPhase 중에 0(서있음)으로 안정 복귀하면 1회 증가
        if (inDownPhase && stableState == STATE_STAND) {
            squatCount++
            inDownPhase = false
            activity?.runOnUiThread {
                fragmentCameraBinding.txtSquatCount.text = squatCount.toString()
                val goalPct = if (squatGoal > 0) (squatCount.toFloat() / squatGoal.toFloat()) * 100f else 0f
                fragmentCameraBinding.txtGoalProgress.text =
                    String.format(java.util.Locale.getDefault(), "%.1f%%", goalPct)
            }
        }
    }

    override fun onError(error: String, errorCode: Int) {
        activity?.runOnUiThread {
            Toast.makeText(requireContext(), error, Toast.LENGTH_SHORT).show()
        }
        if (errorCode == PoseLandmarkerHelper.GPU_ERROR && this::poseLandmarkerHelper.isInitialized) {
            backgroundExecutor.execute {
                try {
                    poseLandmarkerHelper.currentDelegate = PoseLandmarkerHelper.DELEGATE_CPU
                    poseLandmarkerHelper.clearPoseLandmarker()
                    poseLandmarkerHelper.setupPoseLandmarker()
                    Log.i(TAG, "GPU 오류 → CPU delegate로 전환")
                } catch (e: Exception) {
                    Log.e(TAG, "CPU 전환 실패: ${e.message}")
                }
            }
        }
    }

    // --------------------------
    // 각도/로깅 헬퍼
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

    private fun extractPoseLog(
        result: PoseLandmarkerResult,
        imageWidth: Int,
        imageHeight: Int
    ): PoseLog? {
        val lm = result.landmarks().firstOrNull() ?: return null
        if (lm.size <= 28) return null

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

    private fun toFeaturesForTFLite(
        log: PoseLog,
        imgW: Int,
        imgH: Int
    ): FloatArray {
        fun nx(x: Double) = (x / imgW).toFloat()  // 0~1
        fun ny(y: Double) = (y / imgH).toFloat()  // 0~1

        return floatArrayOf(
            nx(log.Rshoulder_x), ny(log.Rshoulder_y),
            nx(log.Lshoulder_x), ny(log.Lshoulder_y),
            nx(log.Rhip_x),      ny(log.Rhip_y),
            nx(log.Lhip_x),      ny(log.Lhip_y),
            nx(log.Rknee_x),     ny(log.Rknee_y),
            nx(log.Lknee_x),     ny(log.Lknee_y),
            nx(log.Rankle_x),    ny(log.Rankle_y),
            nx(log.Lankle_x),    ny(log.Lankle_y),
            log.Rknee_angle.toFloat(),
            log.Lknee_angle.toFloat(),
            log.Rhip_angle.toFloat(),
            log.Lhip_angle.toFloat()
        )
    }
}

