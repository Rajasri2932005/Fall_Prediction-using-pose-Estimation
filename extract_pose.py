import cv2
import csv
import mediapipe as mp
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "pose_landmarker_full.task"
OUTPUT_DIR = BASE_DIR / "Output"

OUTPUT_DIR.mkdir(exist_ok=True)

# Video information and fall intervals
videos = [
    ("FALL", "01.mp4", 3.4, 6.0),
    ("FALL", "05.mp4", 1.8, 5.0),
    ("FALL", "08.mp4", 0.8, 4.0),

    ("ADL", "05.mp4", None, None),
    ("ADL", "08.mp4", None, None),
    ("ADL", "16.mp4", None, None)
]

landmark_names = [
    "nose",
    "left_eye_inner",
    "left_eye",
    "left_eye_outer",
    "right_eye_inner",
    "right_eye",
    "right_eye_outer",
    "left_ear",
    "right_ear",
    "mouth_left",
    "mouth_right",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_pinky",
    "right_pinky",
    "left_index",
    "right_index",
    "left_thumb",
    "right_thumb",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
    "left_heel",
    "right_heel",
    "left_foot_index",
    "right_foot_index"
]

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

def process_video(category, filename, fall_start, fall_end):

    video_path = BASE_DIR / category / filename

    output_name = f"{category}_{Path(filename).stem}_pose_labeled.csv"
    output_path = OUTPUT_DIR / output_name

    print()
    print("=" * 60)
    print("Processing:", category, filename)

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print("ERROR: Could not open video:", video_path)
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print("FPS:", fps)
    print("Total frames:", total_frames)

    if fall_start is not None:
        print("Fall interval:", fall_start, "to", fall_end, "seconds")
    else:
        print("ADL video: No fall interval")

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(
            model_asset_path=str(MODEL_PATH)
        ),
        running_mode=VisionRunningMode.VIDEO,
        num_poses=1
    )

    header = [
        "frame",
        "time",
        "label"
    ]

    for name in landmark_names:
        header.extend([
            f"{name}_x",
            f"{name}_y",
            f"{name}_z",
            f"{name}_visibility"
        ])

    with PoseLandmarker.create_from_options(options) as landmarker:

        with open(output_path, "w", newline="") as csv_file:

            writer = csv.writer(csv_file)
            writer.writerow(header)

            frame_number = 0

            while True:

                success, frame = cap.read()

                if not success:
                    break

                time_seconds = frame_number / fps

                # Label the frame
                if (
                    fall_start is not None
                    and fall_start <= time_seconds <= fall_end
                ):
                    label = 1
                else:
                    label = 0

                # Convert BGR to RGB
                rgb_frame = cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2RGB
                )

                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=rgb_frame
                )

                timestamp_ms = int(time_seconds * 1000)

                result = landmarker.detect_for_video(
                    mp_image,
                    timestamp_ms
                )

                row = [
                    frame_number,
                    time_seconds,
                    label
                ]

                # Store pose landmarks
                if result.pose_landmarks:

                    landmarks = result.pose_landmarks[0]

                    for landmark in landmarks:

                        row.extend([
                            landmark.x,
                            landmark.y,
                            landmark.z,
                            landmark.visibility
                        ])

                else:

                    row.extend([""] * (33 * 4))
                writer.writerow(row)

                frame_number += 1

                if frame_number % 30 == 0:

                    print(
                        f"Processed {frame_number}/{total_frames} frames"
                    )

    cap.release()

    print("Completed:", filename)
    print("Saved:", output_path)

# Process all videos
for category, filename, fall_start, fall_end in videos:

    process_video(
        category,
        filename,
        fall_start,
        fall_end
    )

print()
print("=" * 60)
print("ALL 6 VIDEOS PROCESSED SUCCESSFULLY")
print("=" * 60)
