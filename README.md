# Fall Prediction Using Pose Estimation

Camera-based fall detection: MediaPipe Pose Landmarker (33 landmarks, 132 features) + Random Forest,
with a Windows desktop alert app, a Flask service and an Android app.

## Folder layout
```
python/
  extract_pose.py        # A.1  video -> labelled landmark CSVs (Output/*_pose_labeled.csv)
  combine_dataset.py     # A.2  merge the six CSVs -> Output/combined_pose_dataset.csv
  train_model.py         # A.3  train Random Forest (200 trees, balanced) -> fall_detection_model.pkl
  evaluate_model.py      # A.4  metrics, confusion matrix, plots -> Output/evaluation/
  demo_alert.py          # A.5  desktop video demo with Windows alarm + toast
  app.py                 # A.6  Flask + Waitress API
android/
  OfflineVideoPredictor.kt                 # A.7.3 recorded-video analysis on the phone
  MainActivity_fragments_from_report.kt    # A.7.1-A.7.4 fragments, see note below
requirements.txt
```

## Run order (Python 3.10+)
```
pip install -r requirements.txt
# put pose_landmarker_full.task (from the MediaPipe site) next to the scripts,
# and GMDCSA-24 videos in ADL/ and FALL/ folders
python python/extract_pose.py
python python/combine_dataset.py
# -> create Output/clean_pose_dataset.csv (see "Missing pieces")
python python/train_model.py
python python/evaluate_model.py
python python/demo_alert.py          # Windows only (winsound, winotify)
python python/app.py                 # optional API on port 5000
```

## Missing pieces to add from your own project (not in the report)
1. The step that turns combined_pose_dataset.csv into Output/clean_pose_dataset.csv (removing no-pose frames).
2. FallModel (Kotlin/Java) used by the Android code: FallModel.score(input).
3. Android manifest, Gradle files and the full MainActivity.kt.
   MainActivity_fragments_from_report.kt only joins the snippets printed in the report; copy the real
   MainActivity.kt from Android Studio instead.
4. pose_landmarker_full.task and the dataset are not included (check the licences).

## Before making the repository public
- Remove the leftover `API_BASE_URL = "http://172.22.37.140:5000"` in the Android code (unused, local IP).
- Dataset: GMDCSA-24 v2.1, https://doi.org/10.5281/zenodo.13354453
