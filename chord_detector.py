#!/usr/bin/env python3
"""
Chord detection from audio files using a trained CNN model.
"""
import os
import numpy as np
import librosa
import tensorflow as tf

def load_class_names(base_path="datasets"):
    """Load chord names from dataset folder structure."""
    names = sorted([d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))])
    return names

def get_model_input_shape(model):
    """Infer expected input shape (n_bins, max_len) from model."""
    try:
        ishp = model.input_shape
    except Exception:
        ishp = None
    
    if isinstance(ishp, (list, tuple)):
        if isinstance(ishp[0], (list, tuple)):
            ishp = ishp[0]
    
    if ishp and len(ishp) >= 3 and ishp[1] and ishp[2]:
        return int(ishp[1]), int(ishp[2])
    
    try:
        ishp = model.inputs[0].shape.as_list()
        return int(ishp[1]), int(ishp[2])
    except Exception:
        return None, None

def wav_to_chroma_cqt(path, n_chroma=12, max_len=256, sr=22050, hop_length=512):
    """Convert WAV to Chroma CQT feature."""
    y, _ = librosa.load(path, sr=sr)
    y, _ = librosa.effects.trim(y, top_db=35)
    
    cqt = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop_length, n_chroma=n_chroma)
    eps = 1e-8
    chroma_db = librosa.power_to_db(np.abs(cqt) + eps, ref=np.max)
    
    if chroma_db.shape[1] < max_len:
        chroma_db = np.pad(chroma_db, ((0,0), (0, max_len - chroma_db.shape[1])), 
                          mode='constant', constant_values=(chroma_db.min(),))
    else:
        chroma_db = chroma_db[:, :max_len]
    
    return chroma_db[..., np.newaxis]

def wav_to_melspec(path, n_mels=128, max_len=256, sr=22050, n_fft=2048, hop_length=512):
    """Convert WAV to Mel-spectrogram feature."""
    y, _ = librosa.load(path, sr=sr)
    y, _ = librosa.effects.trim(y, top_db=35)
    
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    
    if mel_db.shape[1] < max_len:
        mel_db = np.pad(mel_db, ((0,0), (0, max_len - mel_db.shape[1])), 
                       mode='constant', constant_values=(mel_db.min(),))
    else:
        mel_db = mel_db[:, :max_len]
    
    return mel_db[..., np.newaxis]

def detect_chords_from_directory(directory, model, class_names, expected_n=None, expected_t=None, verbose=True):
    """
    Detect chords from all WAV files in a directory.
    Returns list of (filename, chord, confidence).
    """
    files = sorted([os.path.join(directory, f) for f in os.listdir(directory) 
                   if f.lower().endswith('.wav')])
    
    if not files:
        return []
    
    results = []
    for filepath in files:
        filename = os.path.basename(filepath)
        if verbose:
            print(f"🎵 Processing: {filename}")
        
        # Choose feature type based on expected input shape
        if expected_n == 12:
            feat = wav_to_chroma_cqt(filepath, n_chroma=12, max_len=expected_t or 256)
        else:
            n_mels = expected_n if expected_n is not None else 128
            feat = wav_to_melspec(filepath, n_mels=n_mels, max_len=expected_t or 256)
        
        x = np.expand_dims(feat, axis=0).astype(np.float32)
        
        try:
            preds = model.predict(x, verbose=0)
        except Exception as e:
            if verbose:
                print(f"  ❌ Prediction failed: {e}")
            continue
        
        idx = int(np.argmax(preds))
        conf = float(preds[0, idx])
        chord = class_names[idx] if idx < len(class_names) else f"idx_{idx}"
        
        if verbose:
            print(f"  ➤ {chord}  (confidence: {conf:.2f})")
        
        results.append((filename, chord, conf))
    
    return results

def load_cnn_model(model_path):
    """Load a saved Keras CNN model."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    return tf.keras.models.load_model(model_path)

if __name__ == "__main__":
    # Quick test
    import sys
    if len(sys.argv) < 2:
        print("Usage: python chord_detector.py <directory> [--model <path>]")
        sys.exit(1)
    
    directory = sys.argv[1]
    model_path = sys.argv[3] if len(sys.argv) > 3 and sys.argv[2] == "--model" else "models/chord_cnn_chroma.keras"
    
    model = load_cnn_model(model_path)
    class_names = load_class_names()
    expected_n, expected_t = get_model_input_shape(model)
    
    results = detect_chords_from_directory(directory, model, class_names, expected_n, expected_t)
    chords = [chord for _, chord, _ in results]
    print("\n🎶 Detected chords:", chords)