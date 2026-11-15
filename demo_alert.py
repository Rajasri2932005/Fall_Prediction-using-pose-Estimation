import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import joblib
import winsound
from winotify import Notification

# Load trained model
model = joblib.load("fall_detection_model.pkl")

# Load feature names used during training
df_columns = pd.read_csv(
    "Output/clean_pose_dataset.csv",
    nrows=0
).columns

pose_cols = df_columns[3:-2]

# MediaPipe Tasks
BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Pose Landmarker settings
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

# Get video path
video_path = input("Enter video path: ")

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)

if fps <= 0:
    fps = 30

frame_number = 0

# Temporal smoothing
fall_counter = 0
FALL_THRESHOLD = 5

# Prevent repeated alerts for the same fall
alert_sent = False

# Create pose landmarker
with PoseLandmarker.create_from_options(options) as landmarker:

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        timestamp_ms = int((frame_number / fps) * 1000)

        result = landmarker.detect_for_video(
            mp_image,
            timestamp_ms
        )

        # Header
        cv2.rectangle(
            frame,
            (0, 0),
            (frame.shape[1], 90),
            (30, 30, 30),
            -1
        )

        cv2.putText(
            frame,
            "FALL PREDICTION SYSTEM",
            (25, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2
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

            features = np.array(features).reshape(1, -1)

            features_df = pd.DataFrame(
                features,
                columns=pose_cols
            )

            prediction = model.predict(features_df)[0]

            if prediction == 1:
                fall_counter += 1
            else:
                fall_counter = 0
                alert_sent = False

            if fall_counter >= FALL_THRESHOLD:

                label = "FALL DETECTED"
                text_color = (0, 0, 255)

                # Send alert only once for the detected fall
                if not alert_sent:

                    # Buzzer / alarm sound
                    # Buzzer / alarm sound - 3 beeps
                    for i in range(3):
                        winsound.Beep(1000, 500)
                         
                    # Windows notification
                    toast = Notification(
                        app_id="Fall Prediction System",
                        title="FALL ALERT!",
                        msg="A possible fall has been detected. Please check the person.",
                        duration="long"
                    )

                    toast.show()

                    alert_sent = True

            else:

                label = "NORMAL"
                text_color = (0, 255, 0)

            cv2.putText(
                frame,
                "Pose: DETECTED",
                (25, 65),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                label,
                (30, 140),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                text_color,
                3
            )

            # Draw pose landmarks
            for landmark in landmarks:

                x = int(landmark.x * frame.shape[1])
                y = int(landmark.y * frame.shape[0])

                cv2.circle(
                    frame,
                    (x, y),
                    4,
                    (255, 255, 255),
                    -1
                )

        else:

            fall_counter = 0
            alert_sent = False

            cv2.putText(
                frame,
                "Pose: NOT DETECTED",
                (25, 65),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 255),
                2
            )

            cv2.putText(
                frame,
                "NO POSE DETECTED",
                (30, 140),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 255),
                2
            )

        cv2.imshow("Fall Prediction System", frame)

        frame_number += 1

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()

print("Video processing completed.")
