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

class CameraFragment : Fragment(), PoseLandmarkerHelper.LandmarkerListener {

    companion object {
        private const val TAG = "Pose Landmarker"
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

    /** Blocking ML operations are performed using this executor */
    private lateinit var backgroundExecutor: ExecutorService

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
            // 현재 값 저장(원하시면 유지/삭제하셔도 됩니다)
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

        // PoseLandmarkerHelper 생성: 임계값=0.5, delegate=CPU, 모델=Lite(인덱스 0)
        backgroundExecutor.execute {
            poseLandmarkerHelper = PoseLandmarkerHelper(
                context = requireContext(),
                runningMode = RunningMode.LIVE_STREAM,
                minPoseDetectionConfidence = 0.5f,  // 🔧 고정
                minPoseTrackingConfidence = 0.5f,   // 🔧 고정
                minPosePresenceConfidence = 0.5f,   // 🔧 고정
                currentDelegate = PoseLandmarkerHelper.DELEGATE_CPU, // 🔧 CPU 기본값
                poseLandmarkerHelperListener = this
            )
            // 🔧 모델 Lite로 고정 (일반적으로 0이 Lite)
            try {
                // 상수가 있다면: poseLandmarkerHelper.currentModel = PoseLandmarkerHelper.MODEL_LITE
                poseLandmarkerHelper.currentModel = 0
                // 모델 반영을 위해 재초기화
                poseLandmarkerHelper.clearPoseLandmarker()
                poseLandmarkerHelper.setupPoseLandmarker()
            } catch (e: Exception) {
                Log.w(TAG, "모델 설정 중 예외 발생: ${e.message}")
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

    // 결과 수신: 바텀시트가 없으므로 텍스트 업데이트 제거, 오버레이만 갱신
    override fun onResults(resultBundle: PoseLandmarkerHelper.ResultBundle) {
        activity?.runOnUiThread {
            if (_fragmentCameraBinding != null) {
                fragmentCameraBinding.overlay.setResults(
                    resultBundle.results.first(),
                    resultBundle.inputImageHeight,
                    resultBundle.inputImageWidth,
                    RunningMode.LIVE_STREAM
                )

                // 각도 계산 로그
                val angles = computeAnglesFrom(
                    resultBundle.results.first(),
                    resultBundle.inputImageWidth,
                    resultBundle.inputImageHeight
                )
                Log.d(
                    TAG,
                    "Angles -> Rknee=${"%.1f".format(angles.Rknee_angle)}, " +
                            "Lknee=${"%.1f".format(angles.Lknee_angle)}, " +
                            "Rhip=${"%.1f".format(angles.Rhip_angle)}, " +
                            "Lhip=${"%.1f".format(angles.Lhip_angle)}"
                )

                fragmentCameraBinding.overlay.invalidate()
            }
        }
    }

    override fun onError(error: String, errorCode: Int) {
        activity?.runOnUiThread {
            Toast.makeText(requireContext(), error, Toast.LENGTH_SHORT).show()
        }
        // 바텀시트가 없으므로 GPU 오류 시 자동으로 CPU로 전환
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
    // ⬇️ 각도 계산 헬퍼
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
}
