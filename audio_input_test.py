# import sounddevice as sd
# from scipy.io.wavfile import write
# import numpy as np
# import librosa
# from joblib import load
# import os
# import time

# # ===========================
# # FEATURE EXTRACTION (same as training)
# # ===========================
# def extract_features(file_path, use_mfcc=False, use_chroma=True, use_spectral=False):
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
#             np.mean(rolloff),
#             np.mean(zcr)
#         ]))

#     return np.hstack(features)

# # ===========================
# # LOAD MODELS
# # ===========================
# rf = load("models/random_forest_model.joblib")
# gb = load("models/gradient_boosting_model.joblib")

# print("Models loaded successfully!")

# # ===========================
# # RECORD AUDIO FROM MIC
# # ===========================
# def record_audio(duration=3, fs=22050, filename="live_input.wav"):
#     print(f"\n🎤 Recording for {duration} seconds... Play your chord now!")
#     time.sleep(1)

#     audio = sd.rec(int(duration * fs), samplerate=fs, channels=1)
#     sd.wait()

#     audio = audio.flatten()
#     write(filename, fs, audio)
#     print("Recording saved as:", filename)

#     return filename

# # ===========================
# # PREDICT CHORD
# # ===========================
# def predict_live_audio():
#     wav_file = record_audio()

#     features = extract_features(wav_file)
#     features = features.reshape(1, -1)

#     rf_pred = rf.predict(features)[0]
#     gb_pred = gb.predict(features)[0]

#     print("\n🎸 ====== LIVE PREDICTION ======")
#     print(f"Random Forest:        {rf_pred}")
#     print(f"Gradient Boosting:    {gb_pred}")
#     print("================================\n")


# if __name__ == "__main__":
#     while True:
#         predict_live_audio()

#         again = input("Record another chord? (y/n): ").strip().lower()
#         if again != "y":
#             break


import sounddevice as sd
print(sd.query_devices())
