// ===== A.7.1 Main activity, imports, state, onCreate, FallPredictionScreen =====
package com.example.fallprediction

import android.Manifest
import android.app.NotificationChannel
import android.app.NotificationManager
import android.content.Context
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Matrix
import android.graphics.Paint
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.view.View
import android.widget.FrameLayout
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Button
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.TextButton
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier

import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.app.NotificationCompat
import androidx.core.app.NotificationManagerCompat
import androidx.core.content.ContextCompat
import androidx.lifecycle.LifecycleOwner
import com.example.fallprediction.ui.theme.FallPredictionTheme
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.framework.image.MPImage
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.poselandmarker.PoseLandmarker
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.MultipartBody
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.asRequestBody
import okhttp3.RequestBody.Companion.toRequestBody
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicInteger
import java.util.concurrent.atomic.AtomicLong
import kotlin.math.max
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll

private const val API_BASE_URL = "http://172.22.37.140:5000"
private const val VIDEO_PREDICT_URL = "$API_BASE_URL/predict"
private const val LIVE_PREDICT_URL = "$API_BASE_URL/predict_pose"

class MainActivity : ComponentActivity() {
    private var selectedVideoUri by mutableStateOf<Uri?>(null)
    private var predictionResult by mutableStateOf("")
    private var isLoading by mutableStateOf(false)
    private var videoProgress by mutableStateOf("Preparing video...")
    private var videoFileName by mutableStateOf("")
    private var predictionTime by mutableStateOf("")

    private var predictionHistory by mutableStateOf(
        emptyList<VideoPredictionHistory>()
    )

    private var showCamera by mutableStateOf(false)

    private var showHistory by mutableStateOf(false)

    private var poseStatus by mutableStateOf("Camera is ready.")
    private var livePredictionStatus by mutableStateOf("Live prediction: waiting for a person...")

    private val liveNetworkExecutor = Executors.newSingleThreadExecutor()
    private val liveRequestInProgress = AtomicBoolean(false)
    private val lastLiveRequestAt = AtomicLong(0L)
    private val consecutiveFallPredictions = AtomicInteger(0)
    private val fallAlertAlreadyShown = AtomicBoolean(false)
    private val livePoseAvailable = AtomicBoolean(false)

    private val httpClient: OkHttpClient by lazy {
        OkHttpClient.Builder()
            .connectTimeout(10, TimeUnit.SECONDS)
            .readTimeout(10, TimeUnit.SECONDS)
            .writeTimeout(10, TimeUnit.SECONDS)
            .build()
    }

    private val videoPicker =
        registerForActivityResult(ActivityResultContracts.GetContent()) { uri ->
            selectedVideoUri = uri
            predictionResult = ""
            predictionTime = ""

            videoFileName = uri?.let {
                getFileName(it)
            } ?: ""
        }

    private val notificationPermissionLauncher =
        registerForActivityResult(ActivityResultContracts.RequestPermission()) { }

    private val cameraPermissionLauncher =
        registerForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
            if (granted) {
                poseStatus = "Starting camera..."
                showCamera = true
            } else {
                predictionResult = "Camera permission was denied. Enable it in app settings."
            }
        }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        predictionHistory = loadPredictionHistory(this)

        createNotificationChannel()
        requestNotificationPermission()

        setContent {
            FallPredictionTheme {
                FallPredictionScreen(
                    selectedVideoUri = selectedVideoUri,
                    predictionResult = predictionResult,
                    isLoading = isLoading,
                    videoProgress = videoProgress,
                    videoFileName = videoFileName,
                    predictionTime = predictionTime,
                    predictionHistory = predictionHistory,

                    showHistory = showHistory,

                    onOpenHistory = {
                        showHistory = true
                    },

                    onCloseHistory = {
                        showHistory = false
                    },

                    onDeleteHistory = {
                        deletePredictionHistory()
                    },

                    showCamera = showCamera,
                    poseStatus = poseStatus,
                    livePredictionStatus = livePredictionStatus,
                    lifecycleOwner = this@MainActivity,
                    onChooseVideo = { videoPicker.launch("video/*") },
                    onOpenCamera = {
                        if (
                            ContextCompat.checkSelfPermission(
                                this@MainActivity,
                                Manifest.permission.CAMERA
                            ) == PackageManager.PERMISSION_GRANTED
                        ) {
                            predictionResult = ""
                            poseStatus = "Starting camera..."
                            livePredictionStatus = "Live prediction: waiting for a person..."
                            consecutiveFallPredictions.set(0)
                            fallAlertAlreadyShown.set(false)
                            showCamera = true
                        } else {
                            cameraPermissionLauncher.launch(Manifest.permission.CAMERA)
                        }
                    },
                    onCloseCamera = {
                        showCamera = false
                        poseStatus = "Camera closed."
                        livePredictionStatus = "Live prediction stopped."
                        consecutiveFallPredictions.set(0)
                        fallAlertAlreadyShown.set(false)
                    },
                    onPoseStatus = { status ->
                        runOnUiThread { poseStatus = status }
                    },
                    onPoseFeatures = { features -> sendLiveFeatures(features) },
                    onDetectFall = { uri ->
                        isLoading = true
                        videoProgress = "Preparing video..."
                        predictionResult = ""
                        predictVideo(uri) { result ->

                            runOnUiThread {
                                predictionResult = result
                                isLoading = false

                                predictionTime = java.text.SimpleDateFormat(
                                    "dd-MM-yyyy HH:mm:ss",
                                    java.util.Locale.getDefault()
                                ).format(java.util.Date())

                                val newHistory = predictionHistory + VideoPredictionHistory(
                                    videoName = videoFileName,
                                    result = result,
                                    time = predictionTime
                                )

                                predictionHistory = newHistory

                                savePredictionHistory(
                                    context = this@MainActivity,
                                    history = newHistory
                                )

                                if (result.equals("FALL DETECTED", ignoreCase = true)) {
                                    showFallNotification()
                                }
                            }

                        }
                    }
                )
            }
        }
    }

@Composable
private fun FallPredictionScreen(
    selectedVideoUri: Uri?,
    predictionResult: String,
    isLoading: Boolean,
    videoProgress: String,
    videoFileName: String,
    predictionTime: String,
    predictionHistory: List<VideoPredictionHistory>,
    showHistory: Boolean,
    onOpenHistory: () -> Unit,
    onCloseHistory: () -> Unit,
    onDeleteHistory: () -> Unit,
    showCamera: Boolean,
    poseStatus: String,
    livePredictionStatus: String,
    lifecycleOwner: LifecycleOwner,
    onChooseVideo: () -> Unit,
    onOpenCamera: () -> Unit,
    onCloseCamera: () -> Unit,
    onPoseStatus: (String) -> Unit,
    onPoseFeatures: (List<Float>) -> Unit,
    onDetectFall: (Uri) -> Unit


) {
    var showDeleteConfirmation by remember {
        mutableStateOf(false)
    }
    Surface(
        modifier = Modifier.fillMaxSize(),
        color = MaterialTheme.colorScheme.background
    ) {
        if (showCamera) {
            Column(
                modifier = Modifier.fillMaxSize().padding(16.dp),
                horizontalAlignment = Alignment.CenterHorizontally
            ) {
                Text(
                    text = "LIVE POSE DETECTION",
                    style = MaterialTheme.typography.headlineSmall,
                    fontWeight = FontWeight.Bold
                )
                Spacer(Modifier.height(8.dp))
                Text(
                    text = poseStatus,
                    textAlign = TextAlign.Center,
                    style = MaterialTheme.typography.bodyLarge
                )
                Spacer(Modifier.height(6.dp))
                Text(
                    text = livePredictionStatus,
                    textAlign = TextAlign.Center,
                    style = MaterialTheme.typography.titleMedium,
                    fontWeight = FontWeight.SemiBold
                )
                Spacer(Modifier.height(12.dp))
                LivePoseCamera(
                    lifecycleOwner = lifecycleOwner,
                    modifier = Modifier.fillMaxWidth().weight(1f),
                    onPoseStatus = onPoseStatus,
                    onPoseFeatures = onPoseFeatures
                )
                Spacer(Modifier.height(12.dp))
                Text(
                    text = "Live classification uses the trained model on your laptop.",
                    textAlign = TextAlign.Center,
                    style = MaterialTheme.typography.bodySmall
                )
                Spacer(Modifier.height(12.dp))
                Button(
                    onClick = onCloseCamera,
                    modifier = Modifier.fillMaxWidth()
                ) {
                    Text("Close Camera")
                }
            }

        } else if (showHistory) {

            Column(
                modifier = Modifier
                    .fillMaxSize()
                    .verticalScroll(rememberScrollState())
                    .padding(24.dp),
                horizontalAlignment = Alignment.CenterHorizontally
            ) {

                Text(
                    text = "Prediction History",
                    style = MaterialTheme.typography.headlineMedium,
                    fontWeight = FontWeight.Bold,
                    textAlign = TextAlign.Center
                )

                Spacer(Modifier.height(20.dp))

                Button(
                    onClick = onCloseHistory,
                    modifier = Modifier.fillMaxWidth()
                ) {
                    Text("Back to Home")
                }

                Spacer(Modifier.height(20.dp))

                if (predictionHistory.isNotEmpty()) {
                    Button(
                        onClick = { showDeleteConfirmation = true },
                        modifier = Modifier.fillMaxWidth()
                    ) {
                        Text("Delete All History")
                    }

                    Spacer(Modifier.height(20.dp))
                }

                if (predictionHistory.isEmpty()) {

                    Text(
                        text = "No prediction history yet.",
                        textAlign = TextAlign.Center
                    )

                } else {

                    predictionHistory.reversed().forEachIndexed { index, history ->

                        Surface(
                            modifier = Modifier
                                .fillMaxWidth()
                                .padding(bottom = 12.dp),
                            color = MaterialTheme.colorScheme.surfaceVariant,
                            shape = MaterialTheme.shapes.medium
                        ) {

                            Column(
                                modifier = Modifier.padding(16.dp)
                            ) {

                                Text(
                                    text = "Prediction ${index + 1}",
                                    style = MaterialTheme.typography.titleMedium,
                                    fontWeight = FontWeight.Bold
                                )

                                Spacer(Modifier.height(8.dp))

                                Text(
                                    text = "Video: ${history.videoName}",
                                    fontWeight = FontWeight.SemiBold
                                )

                                Spacer(Modifier.height(4.dp))

                                Text(
                                    text = "Result: ${history.result}"
                                )

                                Spacer(Modifier.height(4.dp))

                                Text(
                                    text = "Time: ${history.time}",
                                    style = MaterialTheme.typography.bodySmall
                                )
                            }
                        }
                    }
                }
                if (showDeleteConfirmation) {
                    AlertDialog(
                        onDismissRequest = {
                            showDeleteConfirmation = false
                        },
                        title = {
                            Text("Delete All History?")
                        },
                        text = {
                            Text(
                                "Are you sure you want to permanently delete all saved prediction records?"
                            )
                        },
                        confirmButton = {
                            TextButton(
                                onClick = {
                                    onDeleteHistory()
                                    showDeleteConfirmation = false
                                }
                            ) {
                                Text("Delete")
                            }
                        },
                        dismissButton = {
                            TextButton(
                                onClick = {
                                    showDeleteConfirmation = false
                                }
                            ) {
                                Text("Cancel")
                            }
                        }
                    )
                }

            }

        } else {

            Column(
                modifier = Modifier
                    .fillMaxSize()
                    .verticalScroll(rememberScrollState())
                    .padding(24.dp),
                horizontalAlignment = Alignment.CenterHorizontally,
                verticalArrangement = Arrangement.Center
            ) {
                Text(
                    text = "FALL PREDICTION",
                    style = MaterialTheme.typography.headlineMedium,
                    fontWeight = FontWeight.Bold,
                    textAlign = TextAlign.Center
                )
                Spacer(Modifier.height(10.dp))
                Text(
                    text = "Pose Estimation Based Fall Detection",
                    textAlign = TextAlign.Center
                )
                Spacer(Modifier.height(30.dp))
                Button(
                    onClick = onOpenCamera,
                    enabled = !isLoading,
                    modifier = Modifier.fillMaxWidth()
                ) {
                    Text("Open Live Camera")
                }
                Spacer(Modifier.height(12.dp))

                Button(
                    onClick = onOpenHistory,
                    modifier = Modifier.fillMaxWidth()
                ) {
                    Text("Prediction History")
                }
                Spacer(modifier = Modifier.height(16.dp))

                Button(
                    onClick = onChooseVideo,
                    enabled = !isLoading,
                    modifier = Modifier.fillMaxWidth()
                ) {
                    Text("Choose Video")
                }
                if (selectedVideoUri != null) {
                    Spacer(Modifier.height(12.dp))
                    Text("Video selected", fontWeight = FontWeight.Bold)
                    Button(
                        onClick = { onDetectFall(selectedVideoUri) },
                        enabled = !isLoading,
                        modifier = Modifier.fillMaxWidth()
                    ) {
                        Text("Detect Fall")
                    }
                }
                Spacer(Modifier.height(24.dp))

                if (isLoading) {
                    CircularProgressIndicator()

                    Spacer(Modifier.height(12.dp))

                    Text(
                        text = videoProgress,
                        textAlign = TextAlign.Center
                    )
                }else {
                    if (predictionResult.isNotBlank()) {
                        Spacer(Modifier.height(12.dp))

                        Text(
                            text = "Video: $videoFileName",
                            textAlign = TextAlign.Center,
                            style = MaterialTheme.typography.bodyMedium
                        )

                        Spacer(Modifier.height(4.dp))

                        Text(
                            text = "Analyzed on: $predictionTime",
                            textAlign = TextAlign.Center,
                            style = MaterialTheme.typography.bodySmall
                        )

                        Spacer(Modifier.height(12.dp))
                    }
                    when (predictionResult.uppercase()) {
                        "FALL DETECTED" -> {
                            Text(
                                text = "FALL DETECTED",
                                style = MaterialTheme.typography.headlineSmall,
                                fontWeight = FontWeight.Bold
                            )
                            Text("Please check the person immediately.")
                        }

                        "NORMAL" -> {
                            Text(
                                text = "NORMAL",
                                style = MaterialTheme.typography.headlineSmall,
                                fontWeight = FontWeight.Bold
                            )
                            Text("No fall detected.")
                        }

                        else -> {
                            if (predictionResult.isNotBlank()) {
                                Text(
                                    text = predictionResult,
                                    textAlign = TextAlign.Center
                                )
                            }
                        }
                    }

                    Spacer(Modifier.height(24.dp))

                }
            }
        }
    }
}


// ===== A.7.2 Live camera detection =====
private fun sendLiveFeatures(features: List<Float>) {
    // No reliable pose means we cannot make a prediction.
    if (features.isEmpty()) {
        livePoseAvailable.set(false)
        consecutiveFallPredictions.set(0)
        fallAlertAlreadyShown.set(false)

        runOnUiThread {
            livePredictionStatus =
                "UNCERTAIN: Full body not reliably visible. Do not assume the person is safe."
        }
        return
    }

    // Mark the pose as available.
    val poseJustBecameAvailable =
        livePoseAvailable.compareAndSet(false, true)

    if (poseJustBecameAvailable) {
        runOnUiThread {
            livePredictionStatus = "ANALYZING… waiting for prediction"
        }
    }

    // The model requires 33 landmarks x 4 values = 132 features.
    if (features.size != 132) {
        runOnUiThread {
            livePredictionStatus =
                "Live prediction: invalid pose feature count (${features.size})."
        }
        return
    }

    // Limit predictions to approximately two per second.
    val now = System.currentTimeMillis()
    val previous = lastLiveRequestAt.get()

    if (now - previous < 500L) return
    if (!lastLiveRequestAt.compareAndSet(previous, now)) return
    if (!liveRequestInProgress.compareAndSet(false, true)) return

    // Run the model away from the camera/UI thread.
    liveNetworkExecutor.execute {
        try {
            // Convert MediaPipe features to the model's expected input.
            val input = DoubleArray(features.size) { index ->
                val value = features[index]
                if (value.isFinite()) value.toDouble() else 0.0
            }

            // ON-DEVICE CLASSIFICATION: no HTTP request.
            val scores = FallModel.score(input)

            if (scores.size < 2 ||
                !scores[0].isFinite() ||
                !scores[1].isFinite()
            ) {
                throw IllegalStateException("Invalid model scores")
            }

            // Model classes: 0 = Normal, 1 = Fall.
            val prediction = if (scores[1] > scores[0]) 1 else 0
            android.util.Log.d(
                "FallModel",
                "Normal=${scores[0]}, Fall=${scores[1]}, Prediction=$prediction"
            )
            // Ignore stale predictions if the pose disappeared.
            if (!livePoseAvailable.get()) return@execute

            when (prediction) {
                1 -> {
                    val count = consecutiveFallPredictions.incrementAndGet()

                    runOnUiThread {
                        if (!livePoseAvailable.get()) return@runOnUiThread

                        livePredictionStatus = if (count >= 5) {
                            "LIVE PREDICTION: FALL DETECTED"
                        } else {
                            "Live prediction: possible fall ($count/5 confirmations)"
                        }

                        if (
                            count >= 5 &&
                            fallAlertAlreadyShown.compareAndSet(false, true)
                        ) {
                            saveLivePrediction("FALL DETECTED")

                            showFallNotification()
                        }
                    }
                }

                0 -> {
                    consecutiveFallPredictions.set(0)
                    fallAlertAlreadyShown.set(false)

                    runOnUiThread {
                        if (livePoseAvailable.get()) {
                            livePredictionStatus = "Live prediction: NORMAL"
                        }
                    }
                }
            }
        } catch (e: Exception) {
            consecutiveFallPredictions.set(0)

            runOnUiThread {
                if (livePoseAvailable.get()) {
                    livePredictionStatus =
                        "On-device prediction error: ${e.message ?: e.javaClass.simpleName}"
                }
            }
        } finally {
            liveRequestInProgress.set(false)
        }
    }
}

LivePoseCamera(
    lifecycleOwner = lifecycleOwner,
    modifier = Modifier.fillMaxWidth().weight(1f),
    onPoseStatus = onPoseStatus,
    onPoseFeatures = onPoseFeatures
)

@Composable
private fun LivePoseCamera(
    lifecycleOwner: LifecycleOwner,
    modifier: Modifier = Modifier,
    onPoseStatus: (String) -> Unit,
    onPoseFeatures: (List<Float>) -> Unit
) {
    val context = LocalContext.current
    val previewView = remember(context) {
        PreviewView(context).apply {
            scaleType = PreviewView.ScaleType.FIT_CENTER
            implementationMode = PreviewView.ImplementationMode.COMPATIBLE
        }
    }
    val overlayView = remember(context) { PoseOverlayView(context) }
    val container = remember(context) {
        FrameLayout(context).apply {
            addView(
                previewView,
                FrameLayout.LayoutParams(
                    FrameLayout.LayoutParams.MATCH_PARENT,
                    FrameLayout.LayoutParams.MATCH_PARENT
                )
            )
            addView(
                overlayView,
                FrameLayout.LayoutParams(
                    FrameLayout.LayoutParams.MATCH_PARENT,
                    FrameLayout.LayoutParams.MATCH_PARENT
                )
            )
        }
    }

    DisposableEffect(container, lifecycleOwner) {
        val cameraExecutor = Executors.newSingleThreadExecutor()
        val cameraProviderFuture = ProcessCameraProvider.getInstance(context)
        val lastTimestamp = AtomicLong(0L)

        var cameraProvider: ProcessCameraProvider? = null
        var poseLandmarker: PoseLandmarker? = null
        var imageAnalysis: ImageAnalysis? = null

        try {
            val baseOptions = BaseOptions.builder()
                .setModelAssetPath("pose_landmarker_full.task")
                .build()

            val poseOptions = PoseLandmarker.PoseLandmarkerOptions.builder()
                .setBaseOptions(baseOptions)
                .setRunningMode(RunningMode.VIDEO)
                .setNumPoses(1)
                .build()

            poseLandmarker = PoseLandmarker.createFromOptions(context, poseOptions)
        } catch (e: Exception) {
            onPoseStatus("Could not load pose model: ${e.message}")
        }

        cameraProviderFuture.addListener({
            try {
                val provider = cameraProviderFuture.get()
                cameraProvider = provider

                val preview = Preview.Builder().build().also {
                    it.setSurfaceProvider(previewView.surfaceProvider)
                }

                val analysis = ImageAnalysis.Builder()
                    .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                    .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                    .build()

                imageAnalysis = analysis
                analysis.setAnalyzer(cameraExecutor) { imageProxy ->
                    analyzePoseFrame(
                        imageProxy = imageProxy,
                        landmarker = poseLandmarker,
                        lastTimestamp = lastTimestamp,
                        overlayView = overlayView,
                        onPoseStatus = onPoseStatus,
                        onPoseFeatures = onPoseFeatures
                    )
                }

                provider.unbindAll()
                provider.bindToLifecycle(
                    lifecycleOwner,
                    CameraSelector.DEFAULT_BACK_CAMERA,
                    preview,
                    analysis
                )
                onPoseStatus("Camera running. Looking for a person...")
            } catch (e: Exception) {
                onPoseStatus("Could not start camera: ${e.message}")
            }
        }, ContextCompat.getMainExecutor(context))

        onDispose {
            imageAnalysis?.clearAnalyzer()
            cameraProvider?.unbindAll()
            poseLandmarker?.close()
            cameraExecutor.shutdown()
        }
    }

    AndroidView(
        factory = { container },
        modifier = modifier
    )
}

private fun analyzePoseFrame(
    imageProxy: ImageProxy,
    landmarker: PoseLandmarker?,
    lastTimestamp: AtomicLong,
    overlayView: PoseOverlayView,
    onPoseStatus: (String) -> Unit,
    onPoseFeatures: (List<Float>) -> Unit
) {
    var rawBitmap: Bitmap? = null
    var uprightBitmap: Bitmap? = null
    var mpImage: MPImage? = null

    try {
        if (landmarker == null) {
            overlayView.updateLandmarks(emptyList(), 1, 1)
            onPoseStatus("Pose model is not loaded.")
            return
        }

        val plane = imageProxy.planes[0]
        val buffer = plane.buffer.duplicate()
        val width = imageProxy.width
        val height = imageProxy.height
        val rowStride = plane.rowStride
        val pixelStride = plane.pixelStride
        val pixels = IntArray(width * height)

        for (y in 0 until height) {
            for (x in 0 until width) {
                val index = y * rowStride + x * pixelStride
                val r = buffer.get(index).toInt() and 0xFF
                val g = buffer.get(index + 1).toInt() and 0xFF
                val b = buffer.get(index + 2).toInt() and 0xFF
                val a = buffer.get(index + 3).toInt() and 0xFF
                pixels[y * width + x] = Color.argb(a, r, g, b)
            }
        }

        rawBitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
        rawBitmap!!.setPixels(pixels, 0, width, 0, 0, width, height)

        val rotation = imageProxy.imageInfo.rotationDegrees
        uprightBitmap = if (rotation == 0) {
            rawBitmap
        } else {
            val matrix = Matrix().apply { postRotate(rotation.toFloat()) }
            Bitmap.createBitmap(rawBitmap!!, 0, 0, width, height, matrix, true)
        }

        mpImage = BitmapImageBuilder(uprightBitmap!!).build()
        val timestamp = lastTimestamp.updateAndGet { previous ->
            max(System.currentTimeMillis(), previous + 1)
        }

        val result = landmarker.detectForVideo(mpImage!!, timestamp)
        val landmarks = result.landmarks()

        if (landmarks.isNotEmpty()) {
            val personLandmarks = landmarks[0]

            // Filter low-visibility points. This reduces weak detections but cannot
            // guarantee rejection of a clear photograph of a person.
            val reliableLandmarkCount = personLandmarks.count { landmark ->
                landmark.visibility().orElse(0.0f) >= 0.5f
            }

            // For fall detection, require a full-body view, not only a face/shoulders.
            // MediaPipe landmark indices: hips 23/24, knees 25/26, ankles 27/28.
            // This helps reject ID-card headshots, but cannot reject every photograph.
            val requiredLowerBodyIndices = listOf(23, 24, 25, 26, 27, 28)
            // Visibility alone is not enough: inferred joints may lie outside the image
            // when a person is cropped. Require every hip, knee and ankle to be inside
            // a small safety margin from the frame edges as well.
            val lowerBodyVisible = requiredLowerBodyIndices.all { index ->
                if (index >= personLandmarks.size) {
                    false
                } else {
                    val joint = personLandmarks[index]
                    val visibility = joint.visibility().orElse(0.0f)
                    visibility >= 0.5f &&
                            joint.x() in 0.03f..0.97f &&
                            joint.y() in 0.03f..0.97f
                }
            }

            if (reliableLandmarkCount >= 15 && lowerBodyVisible) {
                val points = personLandmarks
                    .filter { landmark -> landmark.visibility().orElse(0.0f) >= 0.5f }
                    .map { landmark -> landmark.x() to landmark.y() }

                overlayView.updateLandmarks(
                    points,
                    uprightBitmap!!.width,
                    uprightBitmap!!.height
                )
                onPoseStatus("FULL-BODY POSE DETECTED\n$reliableLandmarkCount reliable landmarks")

                // Preserve training order: x, y, z, visibility for all 33 landmarks.
                val features: List<Float> = personLandmarks.flatMap { landmark ->
                    listOf<Float>(
                        landmark.x(),
                        landmark.y(),
                        landmark.z(),
                        landmark.visibility().orElse(0.0f)
                    )
                }
                onPoseFeatures(features)
            } else {
                overlayView.updateLandmarks(
                    emptyList(), uprightBitmap!!.width, uprightBitmap!!.height
                )
                onPoseFeatures(emptyList())
                onPoseStatus(
                    if (!lowerBodyVisible) {
                        "UNCERTAIN — FULL BODY NOT VISIBLE"
                    } else {
                        "UNCERTAIN — POSE NOT RELIABLE\nLive prediction paused; check the camera view"
                    }
                )
            }
        } else {
            overlayView.updateLandmarks(
                emptyList(), uprightBitmap!!.width, uprightBitmap!!.height
            )
            onPoseFeatures(emptyList())
            onPoseStatus("UNCERTAIN — NO PERSON POSE DETECTED\nCheck that the person is clearly in frame")
        }
    } catch (e: Exception) {
        overlayView.updateLandmarks(emptyList(), 1, 1)
        onPoseFeatures(emptyList())
        onPoseStatus("UNCERTAIN — POSE DETECTION ERROR\n${e.message ?: "Unknown error"}")
    } finally {
        mpImage?.close()
        if (uprightBitmap !== rawBitmap) uprightBitmap?.recycle()
        rawBitmap?.recycle()
        imageProxy.close()
    }
}

/** Draws pose landmark points aligned to the FIT_CENTER camera image area. */
private class PoseOverlayView(context: Context) : View(context) {
    private val pointPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.GREEN
        style = Paint.Style.FILL
    }

    @Volatile
    private var landmarks: List<Pair<Float, Float>> = emptyList()

    @Volatile
    private var sourceWidth: Int = 1

    @Volatile
    private var sourceHeight: Int = 1

    fun updateLandmarks(points: List<Pair<Float, Float>>, imageWidth: Int, imageHeight: Int) {
        landmarks = points
        sourceWidth = imageWidth.coerceAtLeast(1)
        sourceHeight = imageHeight.coerceAtLeast(1)
        postInvalidate()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        val points = landmarks
        if (points.isEmpty() || width <= 0 || height <= 0) return

        val scale = minOf(width.toFloat() / sourceWidth, height.toFloat() / sourceHeight)
        val drawnWidth = sourceWidth * scale
        val drawnHeight = sourceHeight * scale
        val offsetX = (width - drawnWidth) / 2f
        val offsetY = (height - drawnHeight) / 2f

        for ((x, y) in points) {
            if (x.isFinite() && y.isFinite() && x in 0f..1f && y in 0f..1f) {
                canvas.drawCircle(
                    offsetX + x * drawnWidth,
                    offsetY + y * drawnHeight,
                    5.5f,
                    pointPaint
                )
            }
        }
    }
}


// ===== A.7.3 Recorded-video selection and UI (MainActivity parts) =====
private val videoPicker =
    registerForActivityResult(ActivityResultContracts.GetContent()) { uri ->
        selectedVideoUri = uri
        predictionResult = ""
        predictionTime = ""

        videoFileName = uri?.let {
            getFileName(it)
        } ?: ""
    }

private fun getFileName(uri: Uri): String {
    var fileName = "Selected video"

    val cursor = contentResolver.query(
        uri,
        arrayOf(android.provider.OpenableColumns.DISPLAY_NAME),
        null,
        null,
        null
    )

    cursor?.use {
        if (it.moveToFirst()) {
            val columnIndex = it.getColumnIndex(
                android.provider.OpenableColumns.DISPLAY_NAME
            )

            if (columnIndex >= 0) {
                fileName = it.getString(columnIndex)
                    ?: "Selected video"
            }
        }
    }

    return fileName
}

onDetectFall = { uri ->
    isLoading = true
    videoProgress = "Preparing video..."
    predictionResult = ""
    predictVideo(uri) { result ->

        runOnUiThread {
            predictionResult = result
            isLoading = false

            predictionTime = java.text.SimpleDateFormat(
                "dd-MM-yyyy HH:mm:ss",
                java.util.Locale.getDefault()
            ).format(java.util.Date())

            val newHistory = predictionHistory + VideoPredictionHistory(
                videoName = videoFileName,
                result = result,
                time = predictionTime
            )

            predictionHistory = newHistory

            savePredictionHistory(
                context = this@MainActivity,
                history = newHistory
            )

            if (result.equals("FALL DETECTED", ignoreCase = true)) {
                showFallNotification()
            }
        }

    }
}

private fun predictVideo(uri: Uri, onResult: (String) -> Unit) {
    OfflineVideoPredictor.predict(
        context = this,
        uri = uri,
        onProgress = { message ->
            runOnUiThread {
                videoProgress = message
                android.util.Log.d("VideoPrediction", message)
            }
        },
        onResult = { result ->
            runOnUiThread {
                onResult(result)
            }
        }
    )
}

Button(
    onClick = onChooseVideo,
    enabled = !isLoading,
    modifier = Modifier.fillMaxWidth()
) {
    Text("Choose Video")
}
if (selectedVideoUri != null) {
    Spacer(Modifier.height(12.dp))
    Text("Video selected", fontWeight = FontWeight.Bold)
    Button(
        onClick = { onDetectFall(selectedVideoUri) },
        enabled = !isLoading,
        modifier = Modifier.fillMaxWidth()
    ) {
        Text("Detect Fall")
    }
}
Spacer(Modifier.height(24.dp))

if (isLoading) {
    CircularProgressIndicator()

    Spacer(Modifier.height(12.dp))

    Text(
        text = videoProgress,
        textAlign = TextAlign.Center
    )
}else {
    if (predictionResult.isNotBlank()) {
        Spacer(Modifier.height(12.dp))

        Text(
            text = "Video: $videoFileName",
            textAlign = TextAlign.Center,
            style = MaterialTheme.typography.bodyMedium
        )

        Spacer(Modifier.height(4.dp))

        Text(
            text = "Analyzed on: $predictionTime",
            textAlign = TextAlign.Center,
            style = MaterialTheme.typography.bodySmall
        )

        Spacer(Modifier.height(12.dp))
    }
    when (predictionResult.uppercase()) {
        "FALL DETECTED" -> {
            Text(
                text = "FALL DETECTED",
                style = MaterialTheme.typography.headlineSmall,
                fontWeight = FontWeight.Bold
            )
            Text("Please check the person immediately.")
        }

        "NORMAL" -> {
            Text(
                text = "NORMAL",
                style = MaterialTheme.typography.headlineSmall,
                fontWeight = FontWeight.Bold
            )
            Text("No fall detected.")
        }

        else -> {
            if (predictionResult.isNotBlank()) {
                Text(
                    text = predictionResult,
                    textAlign = TextAlign.Center
                )
            }
        }
    }

    Spacer(Modifier.height(24.dp))

}


// ===== A.7.4 Prediction history and notifications =====
data class VideoPredictionHistory(
    val videoName: String,
    val result: String,
    val time: String
)


private fun savePredictionHistory(
    context: Context,
    history: List<VideoPredictionHistory>
) {
    val jsonArray = JSONArray()

    history.forEach { item ->
        val jsonObject = JSONObject()
        jsonObject.put("videoName", item.videoName)
        jsonObject.put("result", item.result)
        jsonObject.put("time", item.time)

        jsonArray.put(jsonObject)
    }

    val sharedPreferences = context.getSharedPreferences(
        "FallPredictionHistory",
        Context.MODE_PRIVATE
    )

    sharedPreferences.edit()
        .putString("history", jsonArray.toString())
        .apply()
}

private fun loadPredictionHistory(
    context: Context
): List<VideoPredictionHistory> {
    val sharedPreferences = context.getSharedPreferences(
        "FallPredictionHistory",
        Context.MODE_PRIVATE
    )

    val historyJson = sharedPreferences.getString("history", null)
        ?: return emptyList()

    return try {
        val jsonArray = JSONArray(historyJson)
        val historyList = mutableListOf<VideoPredictionHistory>()

        for (i in 0 until jsonArray.length()) {
            val jsonObject = jsonArray.getJSONObject(i)

            historyList.add(
                VideoPredictionHistory(
                    videoName = jsonObject.getString("videoName"),
                    result = jsonObject.getString("result"),
                    time = jsonObject.getString("time")
                )
            )
        }

        historyList
    } catch (e: Exception) {
        emptyList()
    }
}

override fun onCreate(savedInstanceState: Bundle?) {
    super.onCreate(savedInstanceState)

    predictionHistory = loadPredictionHistory(this)

    createNotificationChannel()
    requestNotificationPermission()

when (prediction) {
    1 -> {
        val count = consecutiveFallPredictions.incrementAndGet()

        runOnUiThread {
            if (!livePoseAvailable.get()) return@runOnUiThread

            livePredictionStatus = if (count >= 5) {
                "LIVE PREDICTION: FALL DETECTED"
            } else {
                "Live prediction: possible fall ($count/5 confirmations)"
            }

            if (
                count >= 5 &&
                fallAlertAlreadyShown.compareAndSet(false, true)
            ) {
                saveLivePrediction("FALL DETECTED")

                showFallNotification()
            }
        }

private fun saveLivePrediction(result: String) {
    val currentTime = java.text.SimpleDateFormat(
        "dd-MM-yyyy HH:mm:ss",
        java.util.Locale.getDefault()
    ).format(java.util.Date())

    val newHistory = predictionHistory + VideoPredictionHistory(
        videoName = "Live Camera",
        result = result,
        time = currentTime
    )

    predictionHistory = newHistory

    savePredictionHistory(
        context = this,
        history = newHistory
    )
}

private fun deletePredictionHistory() {
    // Clear the history shown on the screen
    predictionHistory = emptyList()

    // Save the empty history to storage
    savePredictionHistory(
        context = this,
        history = emptyList()
    )
}

private fun requestNotificationPermission() {
    if (
        Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU &&
        checkSelfPermission(Manifest.permission.POST_NOTIFICATIONS) != PackageManager.PERMISSION_GRANTED
    ) {
        notificationPermissionLauncher.launch(Manifest.permission.POST_NOTIFICATIONS)
    }
}

private fun createNotificationChannel() {
    if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
        val channel = NotificationChannel(
            "FALL_ALERT_CHANNEL",
            "Fall Alerts",
            NotificationManager.IMPORTANCE_HIGH
        ).apply {
            description = "Notifications for possible fall detection"
        }
        val manager = getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
        manager.createNotificationChannel(channel)
    }
}

private fun showFallNotification() {
    if (
        Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU &&
        checkSelfPermission(Manifest.permission.POST_NOTIFICATIONS) != PackageManager.PERMISSION_GRANTED
    ) {
        return
    }

    val notification = NotificationCompat.Builder(this, "FALL_ALERT_CHANNEL")
        .setSmallIcon(android.R.drawable.ic_dialog_alert)
        .setContentTitle("FALL ALERT!")
        .setContentText("Possible fall detected. Please check the person.")
        .setPriority(NotificationCompat.PRIORITY_HIGH)
        .setAutoCancel(true)
        .build()

    NotificationManagerCompat.from(this).notify(1001, notification)
}

else if (showHistory) {

    Column(
        modifier = Modifier
            .fillMaxSize()
            .verticalScroll(rememberScrollState())
            .padding(24.dp),
        horizontalAlignment = Alignment.CenterHorizontally
    ) {

        Text(
            text = "Prediction History",
            style = MaterialTheme.typography.headlineMedium,
            fontWeight = FontWeight.Bold,
            textAlign = TextAlign.Center
        )

        Spacer(Modifier.height(20.dp))

        Button(
            onClick = onCloseHistory,
            modifier = Modifier.fillMaxWidth()
        ) {
            Text("Back to Home")
        }

        Spacer(Modifier.height(20.dp))

        if (predictionHistory.isNotEmpty()) {
            Button(
                onClick = { showDeleteConfirmation = true },
                modifier = Modifier.fillMaxWidth()
            ) {
                Text("Delete All History")
            }

            Spacer(Modifier.height(20.dp))
        }

        if (predictionHistory.isEmpty()) {

            Text(
                text = "No prediction history yet.",
                textAlign = TextAlign.Center
            )

        } else {

            predictionHistory.reversed().forEachIndexed { index, history ->

                Surface(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(bottom = 12.dp),
                    color = MaterialTheme.colorScheme.surfaceVariant,
                    shape = MaterialTheme.shapes.medium
                ) {

                    Column(
                        modifier = Modifier.padding(16.dp)
                    ) {

                        Text(
                            text = "Prediction ${index + 1}",
                            style = MaterialTheme.typography.titleMedium,
                            fontWeight = FontWeight.Bold
                        )

                        Spacer(Modifier.height(8.dp))

                        Text(
                            text = "Video: ${history.videoName}",
                            fontWeight = FontWeight.SemiBold
                        )

                        Spacer(Modifier.height(4.dp))

                        Text(
                            text = "Result: ${history.result}"
                        )

                        Spacer(Modifier.height(4.dp))

                        Text(
                            text = "Time: ${history.time}",
                            style = MaterialTheme.typography.bodySmall
                        )
                    }
                }
            }
        }
        if (showDeleteConfirmation) {
            AlertDialog(
                onDismissRequest = {
                    showDeleteConfirmation = false
                },
                title = {
                    Text("Delete All History?")
                },
                text = {
                    Text(
                        "Are you sure you want to permanently delete all saved prediction records?"
                    )
                },
                confirmButton = {
                    TextButton(
                        onClick = {
                            onDeleteHistory()
                            showDeleteConfirmation = false
                        }
                    ) {
                        Text("Delete")
                    }
                },
                dismissButton = {
                    TextButton(
                        onClick = {
                            showDeleteConfirmation = false
                        }
                    ) {
                        Text("Cancel")
                    }
                }
            )
        }

    }


