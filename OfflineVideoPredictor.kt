package com.example.fallprediction
import android.content.Context
import android.graphics.Bitmap
import android.media.MediaMetadataRetriever
import android.net.Uri
import android.util.Log
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.poselandmarker.PoseLandmarker
import java.util.concurrent.Executors

object OfflineVideoPredictor {

    private val executor = Executors.newSingleThreadExecutor()

    fun predict(
        context: Context,
        uri: Uri,
        onProgress: (String) -> Unit,
        onResult: (String) -> Unit
    ) {
        executor.execute {
            var retriever: MediaMetadataRetriever? = null
            var landmarker: PoseLandmarker? = null

            try {
                onProgress("Opening video...")

                retriever = MediaMetadataRetriever()
                retriever.setDataSource(context, uri)

                val durationMs = retriever.extractMetadata(
                    MediaMetadataRetriever.METADATA_KEY_DURATION
                )?.toLongOrNull() ?: 0L

                if (durationMs <= 0L) {
                    throw IllegalArgumentException(
                        "Could not read the video duration."
                    )
                }

                val options = PoseLandmarker.PoseLandmarkerOptions.builder()
                    .setBaseOptions(
                        BaseOptions.builder()
                            .setModelAssetPath("pose_landmarker_full.task")
                            .build()
                    )
                    .setRunningMode(RunningMode.VIDEO)
                    .setNumPoses(1)
                    .build()

                landmarker = PoseLandmarker.createFromOptions(
                    context,
                    options
                )

                var normalFrames = 0
                var fallFrames = 0
                var noPoseFrames = 0
                var consecutiveFalls = 0
                var fallDetected = false

                // Analyse approximately 5 frames per second.
                val intervalMs = 200L
                var timestampMs = 0L

                while (timestampMs < durationMs) {
                    val progressPercent =
                        ((timestampMs * 100) / durationMs)
                            .toInt()
                            .coerceIn(0, 100)

                    onProgress(
                        "Analysing video: $progressPercent%"
                    )

                    val bitmap: Bitmap? = retriever.getFrameAtTime(
                        timestampMs * 1000L,
                        MediaMetadataRetriever.OPTION_CLOSEST
                    )

                    if (bitmap != null) {
                        try {
                            val mpImage = BitmapImageBuilder(bitmap).build()

                            try {
                                val result = landmarker.detectForVideo(
                                    mpImage,
                                    timestampMs
                                )

                                val landmarks = result.landmarks().firstOrNull()

                                if (landmarks == null || landmarks.size < 33) {
                                    noPoseFrames++
                                    consecutiveFalls = 0
                                } else {
                                    val features = mutableListOf<Float>()

                                    for (point in landmarks.take(33)) {
                                        features.add(point.x())
                                        features.add(point.y())
                                        features.add(point.z())
                                        features.add(
                                            point.visibility().orElse(0.0f)
                                        )
                                    }

                                    val validPose = landmarks.count {
                                        it.visibility().orElse(0.0f) >= 0.5f
                                    } >= 15

                                    if (!validPose || features.size != 132) {
                                        noPoseFrames++
                                        consecutiveFalls = 0
                                    } else {
                                        val input = DoubleArray(132) { i ->
                                            val value = features[i]
                                            if (value.isFinite()) {
                                                value.toDouble()
                                            } else {
                                                0.0
                                            }
                                        }

                                        val scores = FallModel.score(input)

                                        if (
                                            scores.size < 2 ||
                                            !scores[0].isFinite() ||
                                            !scores[1].isFinite()
                                        ) {
                                            throw IllegalStateException(
                                                "Invalid model output."
                                            )
                                        }

                                        val prediction =
                                            if (scores[1] > scores[0]) 1 else 0

                                        if (prediction == 1) {
                                            fallFrames++
                                            consecutiveFalls++
                                            if (consecutiveFalls >= 5) {
                                                fallDetected = true
                                            }
                                        } else {
                                            normalFrames++
                                            consecutiveFalls = 0
                                        }
                                    }
                                }
                            } finally {
                                mpImage.close()
                            }
                        } finally {
                            bitmap.recycle()
                        }
                    } else {
                        noPoseFrames++
                        consecutiveFalls = 0
                    }

                    timestampMs += intervalMs
                }

                val message = when {
                    fallDetected ->
                        "FALL DETECTED\n\n" +
                                "Fall predictions: $fallFrames\n" +
                                "Normal predictions: $normalFrames\n" +
                                "Frames without usable pose: $noPoseFrames"

                    normalFrames > 0 ->
                        "NORMAL\n\n" +
                                "Normal predictions: $normalFrames\n" +
                                "Fall predictions: $fallFrames\n" +
                                "Frames without usable pose: $noPoseFrames"

                    else ->
                        "NO POSE DETECTED\n\n" +
                                "The app could not obtain enough usable body poses."
                }

                Log.d("OfflineVideoPredictor", message)
                onResult(message)

            } catch (e: Exception) {
                Log.e("OfflineVideoPredictor", "Video analysis failed", e)
                onResult("Video analysis failed: ${e.localizedMessage ?: "Unknown error"}")
            } finally {
                landmarker?.close()
                retriever?.release()
            }
        }
    }
}
