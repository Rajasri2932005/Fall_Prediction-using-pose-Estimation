import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report
)

# Load dataset
df = pd.read_csv("Output/clean_pose_dataset.csv")

# Pose features
pose_cols = df.columns[3:-2]

# Same video-level split used during training
train_videos = ["ADL_05", "ADL_08", "FALL_01", "FALL_05"]
test_videos = ["ADL_16", "FALL_08"]

test_df = df[df["video"].isin(test_videos)].copy()

X_test = test_df[pose_cols]
y_test = test_df["label"]

# Load trained model
model = joblib.load("fall_detection_model.pkl")

# Predictions
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("========== MODEL EVALUATION ==========")
print("Testing videos:", test_videos)
print("Testing samples:", len(X_test))
print("Accuracy:", round(accuracy * 100, 2), "%")

# Classification report
print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=["Normal", "Fall"]
    )
)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

# Create output folder for graphs
import os
os.makedirs("Output/evaluation", exist_ok=True)

# Confusion Matrix Graph
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Normal", "Fall"]
)

disp.plot()

plt.title("Fall Detection - Confusion Matrix")
plt.tight_layout()

plt.savefig(
    "Output/evaluation/confusion_matrix.png",
    dpi=300
)

plt.show()

print("\nConfusion matrix saved:")
print("Output/evaluation/confusion_matrix.png")

# Metrics
report = classification_report(
    y_test,
    y_pred,
    target_names=["Normal", "Fall"],
    output_dict=True
)

metrics = {
    "Accuracy": accuracy,
    "Normal Precision": report["Normal"]["precision"],
    "Normal Recall": report["Normal"]["recall"],
    "Normal F1-score": report["Normal"]["f1-score"],
    "Fall Precision": report["Fall"]["precision"],
    "Fall Recall": report["Fall"]["recall"],
    "Fall F1-score": report["Fall"]["f1-score"]
}

# Metrics graph
names = list(metrics.keys())
values = list(metrics.values())

plt.figure(figsize=(10, 6))

plt.bar(names, values)

plt.ylabel("Score")
plt.title("Fall Detection Model Performance")
plt.ylim(0, 1)

plt.xticks(
    rotation=30,
    ha="right"
)

plt.tight_layout()

plt.savefig(
    "Output/evaluation/model_performance.png",
    dpi=300
)

plt.show()

print("Performance graph saved:")
print("Output/evaluation/model_performance.png")

print("\nEvaluation completed successfully.")
