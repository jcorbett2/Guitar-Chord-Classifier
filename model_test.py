# import librosa
# import numpy as np
# from joblib import load
# import os

# def extract_features(file_path, use_mfcc=True, use_chroma=True, use_spectral=False):
#     y, sr = librosa.load(file_path, sr=22050)

#     stft = np.abs(librosa.stft(y)) if use_chroma or use_spectral else None
    
#     features = []

#     if use_mfcc:
#         mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
#         features.append(np.mean(mfccs, axis=1))
    
#     if use_chroma:
#         chroma = librosa.feature.chroma_stft(S=stft, sr=sr)
#         features.append(np.mean(chroma, axis=1))
    
#     if use_spectral:
#         centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
#         bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
#         rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
#         zcr = librosa.feature.zero_crossing_rate(y)
#         features.append(np.array([
#             np.mean(centroid),
#             np.mean(bandwidth),
#             np.mean(rolloff),
#             np.mean(zcr)
#         ]))

#     return np.hstack(features)

# # Load dataset
# def load_dataset(base_path="datasets"):
#     X, y = [], []

#     for label in os.listdir(base_path):
#         class_dir = os.path.join(base_path, label)
#         if not os.path.isdir(class_dir):
#             continue

#         for file in os.listdir(class_dir):
#             if file.lower().endswith(".wav"):
#                 file_path = os.path.join(class_dir, file)
#                 features = extract_features(file_path, use_mfcc=True, use_chroma=True)
#                 X.append(features)
#                 y.append(label)


#     ##tests the number of audio clips each folder has
#     # from collections import Counter
#     # print("Number of samples loaded:", len(X))
#     # print("Class distribution:", Counter(y))

#     return np.array(X), np.array(y)


# # ---- LOAD MODELS ----
# rf = load("models/random_forest_model.joblib")
# gb = load("models/gradient_boosting_model.joblib")

# print("Models loaded successfully!")

# # ---- PREDICT FUNCTION ----
# def predict_chord(file_path):
#     if not os.path.exists(file_path):
#         print("File does not exist:", file_path)
#         return

#     features = extract_features(file_path)
#     features = features.reshape(1, -1)

#     rf_pred = rf.predict(features)[0]
#     gb_pred = gb.predict(features)[0]

#     print("\n--- PREDICTION RESULTS ---")
#     print(f"Random Forest:        {rf_pred}")
#     print(f"Gradient Boosting:    {gb_pred}")
#     print("-------------------------\n")


# # ---- MAIN ----
# if __name__ == "__main__":
#     import sys
#     if len(sys.argv) < 2:
#         print("Usage: python model_test.py path/to/audio.wav")
#         exit()

#     audio_path = sys.argv[1]
#     predict_chord(audio_path)


import librosa
import numpy as np
from joblib import load
import os
import random

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


# ---- LOAD MODELS ----
rf = load("models/random_forest_model.joblib")
gb = load("models/gradient_boosting_model.joblib")

print("Models loaded successfully!")


# ---- PREDICT FUNCTION ----
def predict_chord(file_path):
    features = extract_features(file_path)
    features = features.reshape(1, -1)

    rf_pred = rf.predict(features)[0]
    gb_pred = gb.predict(features)[0]

    print("\n--- PREDICTION RESULTS ---")
    print(f"Random Forest:        {rf_pred}")
    print(f"Gradient Boosting:    {gb_pred}")
    print("-------------------------\n")


# ---- NEW FUNCTION: Pick random .wav per folder ----
def get_random_wav_files(base_path="datasets"):
    selected = []
    
    for folder in os.listdir(base_path):
        folder_path = os.path.join(base_path, folder)
        if not os.path.isdir(folder_path):
            continue
        
        wavs = [f for f in os.listdir(folder_path) if f.lower().endswith(".wav")]
        if len(wavs) == 0:
            print(f"⚠️ No WAV files in folder: {folder}")
            continue
        
        chosen = random.choice(wavs)
        selected.append((folder, os.path.join(folder_path, chosen)))

    return selected


# ---- MAIN ----
if __name__ == "__main__":
    print("\n📁 Searching dataset and selecting random test files...\n")
    test_files = get_random_wav_files()

    for true_label, file_path in test_files:
        print(f"🎵 Testing file: {file_path}")
        print(f"   True Label: {true_label}")
        predict_chord(file_path)
