#!/usr/bin/env python3
import os
import sys
import argparse
import numpy as np
import librosa
import tensorflow as tf

def load_class_names(base_path="datasets"):
    names = sorted([d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))])
    return names

def get_model_input_shape(m):
    # Try common model.input_shape patterns
    try:
        ishp = m.input_shape
    except Exception:
        ishp = None
    if isinstance(ishp, (list, tuple)):
        if isinstance(ishp[0], (list, tuple)):
            ishp = ishp[0]
    if ishp and len(ishp) >= 3 and ishp[1] and ishp[2]:
        return int(ishp[1]), int(ishp[2])
    # fallback from inputs
    try:
        ishp = m.inputs[0].shape.as_list()
        return int(ishp[1]), int(ishp[2])
    except Exception:
        return None, None

def wav_to_chroma_cqt(path, n_chroma=12, max_len=256, sr=22050, hop_length=512):
    y, _ = librosa.load(path, sr=sr)
    y, _ = librosa.effects.trim(y, top_db=35)
    cqt = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop_length, n_chroma=n_chroma)
    eps = 1e-8
    chroma_db = librosa.power_to_db(np.abs(cqt) + eps, ref=np.max)
    if chroma_db.shape[1] < max_len:
        chroma_db = np.pad(chroma_db, ((0,0), (0, max_len - chroma_db.shape[1])), mode='constant', constant_values=(chroma_db.min(),))
    else:
        chroma_db = chroma_db[:, :max_len]
    return chroma_db[..., np.newaxis]

def wav_to_melspec(path, n_mels=128, max_len=256, sr=22050, n_fft=2048, hop_length=512):
    y, _ = librosa.load(path, sr=sr)
    y, _ = librosa.effects.trim(y, top_db=35)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    if mel_db.shape[1] < max_len:
        mel_db = np.pad(mel_db, ((0,0), (0, max_len - mel_db.shape[1])), mode='constant', constant_values=(mel_db.min(),))
    else:
        mel_db = mel_db[:, :max_len]
    return mel_db[..., np.newaxis]

def predict_path(path, model, class_names, expected_n=None, expected_t=None):
    files = []
    if os.path.isdir(path):
        files = [os.path.join(path, f) for f in os.listdir(path) if f.lower().endswith('.wav')]
        files.sort()
    elif os.path.isfile(path) and path.lower().endswith('.wav'):
        files = [path]
    else:
        print("No wav files found at path:", path)
        return []

    results = []
    for f in files:
        print("Processing:", f)
        # choose chroma if expected_n == 12, else fallback to mel
        if expected_n == 12:
            feat = wav_to_chroma_cqt(f, n_chroma=12, max_len=expected_t or 256)
        else:
            # use mel default n_mels=expected_n or 128
            n_mels = expected_n if expected_n is not None else 128
            feat = wav_to_melspec(f, n_mels=n_mels, max_len=expected_t or 256)
        x = np.expand_dims(feat, axis=0).astype(np.float32)
        try:
            preds = model.predict(x, verbose=0)
        except Exception as e:
            print("Prediction error:", e)
            print("feature shape:", x.shape, "expected:", (None, expected_n, expected_t, 1))
            continue
        idx = int(np.argmax(preds))
        conf = float(preds[0, idx])
        label = class_names[idx] if idx < len(class_names) else f"idx_{idx}"
        print(f"  -> {label}  (conf: {conf:.2f})")
        results.append((os.path.basename(f), label, conf))
    return results

def main():
    p = argparse.ArgumentParser()
    p.add_argument("path", help="wav file or directory")
    p.add_argument("--model", help="model path", default="models/chord_cnn_chroma.keras")
    p.add_argument("--data-dir", default="datasets")
    args = p.parse_args()

    if not os.path.exists(args.model):
        print("Model not found:", args.model)
        sys.exit(1)

    model = tf.keras.models.load_model(args.model)
    class_names = load_class_names(args.data_dir)

    expected_n, expected_t = get_model_input_shape(model)
    print("Model expected (n_bins, time_frames) =", expected_n, expected_t)

    results = predict_path(args.path, model, class_names, expected_n, expected_t)
    if results:
        print("\nResults:")
        for fn, lab, conf in results:
            print("===========================")
            print(f"File: {fn}")
            print(f"Chord Prediction: {lab}")
            print(f"Accuracy: {conf*100:.1f}%")
            print("===========================\n\n")


if __name__ == "__main__":
    main()