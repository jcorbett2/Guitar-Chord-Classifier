import librosa
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

#Extract features
def extract_features(file_path, use_mfcc=True, use_chroma=True, use_spectral=False):
    y, sr = librosa.load(file_path, sr=22050)

    stft = np.abs(librosa.stft(y)) if use_chroma or use_spectral else None
    
    features = []

    if use_mfcc:
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        features.append(np.mean(mfccs, axis=1))
    
    if use_chroma:
        chroma = librosa.feature.chroma_stft(S=stft, sr=sr)
        features.append(np.mean(chroma, axis=1))
    
    if use_spectral:
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        zcr = librosa.feature.zero_crossing_rate(y)
        features.append(np.array([
            np.mean(centroid),
            np.mean(bandwidth),
            np.mean(rolloff),
            np.mean(zcr)
        ]))

    return np.hstack(features)

# Load dataset
def load_dataset(base_path="datasets"):
    X, y = [], []

    for label in os.listdir(base_path):
        class_dir = os.path.join(base_path, label)
        if not os.path.isdir(class_dir):
            continue

        for file in os.listdir(class_dir):
            if file.lower().endswith(".wav"):
                file_path = os.path.join(class_dir, file)
                features = extract_features(file_path, use_mfcc=True, use_chroma=True)
                X.append(features)
                y.append(label)


    ##tests the number of audio clips each folder has
    # from collections import Counter
    # print("Number of samples loaded:", len(X))
    # print("Class distribution:", Counter(y))

    return np.array(X), np.array(y)





#Train-test split
X, y = load_dataset("datasets")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42, stratify=y
)

# Random Forest
rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    random_state=42
)
rf.fit(X_train, y_train)

# Gradient Boosting
gb = GradientBoostingClassifier(
    n_estimators=300,
    learning_rate=0.05,
    random_state=42
)
gb.fit(X_train, y_train)


##store models for later use
from joblib import dump

# Create models folder if it doesn't exist
os.makedirs("models", exist_ok=True)

# Save both models
dump(rf, "models/random_forest_model.joblib")
dump(gb, "models/gradient_boosting_model.joblib")

print("Models saved successfully!")
