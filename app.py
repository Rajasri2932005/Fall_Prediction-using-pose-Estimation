import os
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import joblib

from flask import Flask, request, jsonify
from flask_cors import CORS

# -----------------------------
# Flask application
# -----------------------------
app = Flask(__name__)
CORS(app)

# -----------------------------
# Load trained Random Forest
# -----------------------------
model = joblib.load("fall_detection_model.pkl")

# -----------------------------
# Load feature names
# -----------------------------
df_columns = pd.read_csv(
    "Output/clean_pose_dataset.csv",
    nrows=0
).columns

pose_cols = df_columns[3:-2]

# -----------------------------
# MediaPipe Pose Landmarker
# -----------------------------
BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = PoseLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path="pose_landmarker_full.task"
    ),
    running_mode=VisionRunningMode.VIDEO,
    num_poses=1,
    min_pose_detection_confidence=0.5,
    min_pose_presence_confidence=0.5,
    min_tracking_confidence=0.5
)

# -----------------------------
# Prediction function
# -----------------------------
def predict_video(video_path):

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return {
            "success": False,
            "message": "Could not open video."
        }

    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        fps = 30

    frame_number = 0

    fall_counter = 0
    FALL_THRESHOLD = 5

    fall_detected = False
    normal_frames = 0
    fall_frames = 0
    no_pose_frames = 0

    with PoseLandmarker.create_from_options(options) as landmarker:

        while True:

            ret, frame = cap.read()

            if not ret:
                break

            rgb_frame = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb_frame
            )

            timestamp_ms = int(
                (frame_number / fps) * 1000
            )

            result = landmarker.detect_for_video(
                mp_image,
                timestamp_ms
            )

            if result.pose_landmarks:

                landmarks = result.pose_landmarks[0]

                features = []

                for landmark in landmarks:

                    features.extend([
                        landmark.x,
                        landmark.y,
                        landmark.z,
                        landmark.visibility
                    ])

                features = np.array(
                    features
                ).reshape(1, -1)

                features_df = pd.DataFrame(
                    features,
                    columns=pose_cols
                )

                prediction = model.predict(
                    features_df
                )[0]

                if prediction == 1:

                    fall_counter += 1
                    fall_frames += 1

                else:

                    fall_counter = 0
                    normal_frames += 1

                # Five consecutive fall predictions
                if fall_counter >= FALL_THRESHOLD:
                    fall_detected = True

            else:

                fall_counter = 0
                no_pose_frames += 1

            frame_number += 1

    cap.release()

    # -----------------------------
    # Final result
    # -----------------------------
    if fall_detected:

        final_status = "FALL DETECTED"

    elif normal_frames > 0:

        final_status = "NORMAL"

    else:

        final_status = "NO POSE DETECTED"

    return {
        "success": True,
        "status": final_status,
        "total_frames": frame_number,
        "normal_frames": normal_frames,
        "fall_frames": fall_frames,
        "no_pose_frames": no_pose_frames
    }


# -----------------------------
# Home page
# -----------------------------
@app.route("/", methods=["GET"])
def home():

    return jsonify({
        "message": "Fall Prediction API is running",
        "status": "online"
    })


# -----------------------------
# Video prediction API
# -----------------------------
@app.route("/predict", methods=["POST"])
def predict():

    if "video" not in request.files:

        return jsonify({
            "success": False,
            "message": "No video uploaded."
        }), 400

    video = request.files["video"]

    if video.filename == "":
        return jsonify({
            "success": False,
            "message": "No video selected."
        }), 400

    # Temporary file
    temp_path = "uploaded_video.mp4"

    video.save(temp_path)

    try:

        result = predict_video(temp_path)

    finally:

        if os.path.exists(temp_path):
            os.remove(temp_path)

    return jsonify(result)

# -----------------------------
# Live camera pose prediction API
# -----------------------------
@app.route("/predict_pose", methods=["POST"])
def predict_pose():

    data = request.get_json(silent=True)

    if not data or "features" not in data:
        return jsonify({
            "success": False,
            "message": "Missing pose features."
        }), 400

    features = data["features"]

    if not isinstance(features, list) or len(features) != len(pose_cols):
        return jsonify({
            "success": False,
            "message": f"Expected {len(pose_cols)} pose features."
        }), 400

    try:
        values = np.asarray(features, dtype=np.float64)

        if not np.isfinite(values).all():
            raise ValueError("Features must be finite numbers.")

        features_df = pd.DataFrame(
            [values],
            columns=pose_cols
        )

        prediction = int(model.predict(features_df)[0])

        return jsonify({
            "success": True,
            "prediction": prediction,
            "status": "FALL DETECTED" if prediction == 1 else "NORMAL"
        })

    except (ValueError, TypeError) as error:
        return jsonify({
            "success": False,
            "message": str(error)
        }), 400


if __name__ == "__main__":

    print("===================================")
    print(" FALL PREDICTION MOBILE API")
    print("===================================")
    print("Server starting...")
    print("")

    from waitress import serve

    serve(app, host="0.0.0.0", port=5000)
