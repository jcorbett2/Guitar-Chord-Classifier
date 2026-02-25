#!/usr/bin/env python3
import os
import argparse
import numpy as np
import librosa
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras import layers, models, regularizers, callbacks

def augment_audio(y, sr):
    augmented = []
    augmented.append(y)  # original

    # small time-stretch (safe)
    rate = np.random.uniform(0.96, 1.04)
    try:
        augmented.append(librosa.effects.time_stretch(y, rate=rate))
    except Exception:
        augmented.append(y)

    # add light gaussian noise
    noise_amp = 0.004 * np.random.uniform(0.7, 1.3)
    augmented.append(y + noise_amp * np.random.randn(len(y)))

    # random gain
    gain = np.random.uniform(0.85, 1.15)
    augmented.append(y * gain)

    return augmented

def wav_to_chroma_cqt(y, sr, n_chroma=12, max_len=256, hop_length=512):
    # Trim silence
    y, _ = librosa.effects.trim(y, top_db=35)

    # compute chroma CQT
    cqt_chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop_length, n_chroma=n_chroma)

    # Normalize and convert to dB-like scale for stability
    # `cqt_chroma` is already an energy-like measure; convert to dB for dynamic range
    # add small eps to avoid log(0)
    eps = 1e-8
    chroma_db = librosa.power_to_db(np.abs(cqt_chroma) + eps, ref=np.max)

    # pad/trim time axis
    if chroma_db.shape[1] < max_len:
        pad = max_len - chroma_db.shape[1]
        chroma_db = np.pad(chroma_db, ((0, 0), (0, pad)), mode='constant', constant_values=(chroma_db.min(),))
    else:
        chroma_db = chroma_db[:, :max_len]

    return chroma_db

def load_dataset(base_path="datasets", n_chroma=12, max_len=256, sr=22050, augment=True):
    X, y = [], []
    class_names = sorted([d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))])

    if not class_names:
        raise RuntimeError(f"No class directories found in {base_path}")

    for label in class_names:
        class_dir = os.path.join(base_path, label)
        for file in os.listdir(class_dir):
            if not file.lower().endswith(".wav"):
                continue
            file_path = os.path.join(class_dir, file)
            y_raw, _sr = librosa.load(file_path, sr=sr)
            audios = augment_audio(y_raw, sr) if augment else [y_raw]
            for audio in audios:
                chroma = wav_to_chroma_cqt(audio, sr, n_chroma=n_chroma, max_len=max_len)
                X.append(chroma)
                y.append(label)

    X = np.array(X)[..., np.newaxis].astype(np.float32)  # shape (N, 12, max_len, 1)
    y = np.array(y)
    return X, y, class_names

def build_cnn(num_classes, input_shape=(12,256,1), l2=1e-4):
    inp = layers.Input(shape=input_shape)
    x = layers.Conv2D(32, (3,3), padding='same', activation='relu', kernel_regularizer=regularizers.l2(l2))(inp)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2,2))(x)

    x = layers.Conv2D(64, (3,3), padding='same', activation='relu', kernel_regularizer=regularizers.l2(l2))(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2,2))(x)

    # keep chroma dimension small: pool more along time than chroma
    x = layers.Conv2D(128, (3,3), padding='same', activation='relu', kernel_regularizer=regularizers.l2(l2))(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((1,2))(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.4)(x)
    out = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs=inp, outputs=out)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def main(args):
    print("Loading dataset...")
    X, y, class_names = load_dataset(base_path=args.data_dir,
                                     n_chroma=args.n_chroma,
                                     max_len=args.max_len,
                                     sr=args.sr,
                                     augment=args.augment)
    print("Data shapes:", X.shape, y.shape)
    label_to_idx = {label:i for i,label in enumerate(class_names)}
    y_idx = np.array([label_to_idx[l] for l in y])

    X_train, X_test, y_train, y_test = train_test_split(X, y_idx, test_size=args.test_size, stratify=y_idx, random_state=42)

    # compute class weights
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    cw = {i: w for i, w in enumerate(class_weights)}
    print("Class weights:", cw)

    model = build_cnn(num_classes=len(class_names), input_shape=(args.n_chroma, args.max_len, 1))
    model.summary()

    os.makedirs(args.model_dir, exist_ok=True)
    ckpt_path = os.path.join(args.model_dir, args.model_name)

    cbs = [
        callbacks.ModelCheckpoint(ckpt_path, monitor='val_loss', save_best_only=True, verbose=1),
        callbacks.EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True, verbose=1),
        callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, verbose=1)
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=args.epochs,
        batch_size=args.batch_size,
        class_weight=cw,
        callbacks=cbs,
        shuffle=True
    )

    # final evaluation
    print("\nEvaluating on test set...")
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test loss: {loss:.4f}  Test acc: {acc:.4f}")

    # predictions and report
    preds = model.predict(X_test, verbose=0)
    y_pred = np.argmax(preds, axis=1)
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, target_names=class_names, zero_division=0))

    print("\nConfusion matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Save final (best model already saved by checkpoint)
    final_path = os.path.join(args.model_dir, args.model_name.replace('.keras', '_final.keras'))
    model.save(final_path)
    print("Model saved to:", final_path)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=str, default="datasets")
    p.add_argument("--model-dir", type=str, default="models")
    p.add_argument("--model-name", type=str, default="chord_cnn_chroma.keras")
    p.add_argument("--n-chroma", type=int, default=12)
    p.add_argument("--max-len", type=int, default=256)
    p.add_argument("--sr", type=int, default=22050)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--augment", type=bool, default=True)
    args = p.parse_args()
    main(args)