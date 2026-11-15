import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# Load clean dataset
df = pd.read_csv("Output/clean_pose_dataset.csv")

# Pose feature columns
pose_cols = df.columns[3:-2]

# Training and testing videos
train_videos = ["ADL_05", "ADL_08", "FALL_01", "FALL_05"]
test_videos = ["ADL_16", "FALL_08"]

# Split dataset by video
train_df = df[df["video"].isin(train_videos)].copy()
test_df = df[df["video"].isin(test_videos)].copy()

# Features and labels
X_train = train_df[pose_cols]
y_train = train_df["label"]

X_test = test_df[pose_cols]
y_test = test_df["label"]

print("Training videos:", train_videos)
print("Testing videos:", test_videos)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

print("\nTraining labels:")
print(y_train.value_counts())

print("\nTesting labels:")
print(y_test.value_counts())

# Create Random Forest model
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced"
)

# Train
print("\nTraining model...")
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluation
accuracy = accuracy_score(y_test, y_pred)
print("\n========== MODEL RESULTS ==========")
print("Accuracy:", accuracy)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Normal", "Fall"]))

# Save model
joblib.dump(model, "fall_detection_model.pkl")

print("\nModel saved as: fall_detection_model.pkl")
