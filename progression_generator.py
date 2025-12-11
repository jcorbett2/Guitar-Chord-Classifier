#!/usr/bin/env python3
"""
Chord progression generation using LSTM model.
"""
import os
import numpy as np
import tensorflow as tf

ROMAN_TO_IDX = {
    "I": 0, "ii": 1, "iii": 2, "IV": 3, "V": 4, "vi": 5, "vii°": 6
}

IDX_TO_ROMAN = {
    0: "I", 1: "ii", 2: "iii", 3: "IV", 4: "V", 5: "vi", 6: "vii°"
}

def load_lstm_model(model_path):
    """Load trained LSTM model."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"LSTM model not found: {model_path}")
    return tf.keras.models.load_model(model_path)

def encode_roman_numerals(roman_sequence):
    """Convert Roman numerals to integer indices."""
    return [ROMAN_TO_IDX.get(rn, 0) for rn in roman_sequence if rn in ROMAN_TO_IDX]

def decode_prediction(idx):
    """Convert integer index to Roman numeral."""
    return IDX_TO_ROMAN.get(int(idx), "?")

def generate_progression(roman_sequence, lstm_model, num_steps=1, temperature=1.0):
    """
    Generate next chord(s) in progression using LSTM.
    
    Args:
        roman_sequence: list of Roman numerals (e.g., ["I", "IV", "V"])
        lstm_model: loaded LSTM model
        num_steps: how many chords to generate
        temperature: controls randomness (higher = more random)
    
    Returns:
        list of generated Roman numerals
    """
    encoded = encode_roman_numerals(roman_sequence)
    if not encoded:
        return []
    
    generated = []
    current_seq = encoded[-1:]  # use last chord as context
    
    for _ in range(num_steps):
        X = np.array(current_seq)[np.newaxis, :, np.newaxis].astype(np.float32)
        
        preds = lstm_model.predict(X, verbose=0)
        
        # Apply temperature for randomness
        preds = np.log(preds + 1e-8) / temperature
        preds = np.exp(preds) / np.sum(np.exp(preds))
        
        # Sample from distribution (or just take argmax for deterministic)
        next_idx = np.argmax(preds)
        
        generated.append(decode_prediction(next_idx))
        current_seq = [next_idx]
    
    return generated

if __name__ == "__main__":
    # Quick test
    test_roman = ["I", "IV", "V", "I"]
    print(f"Input progression: {test_roman}")
    
    try:
        lstm = load_lstm_model("models/chord_lstm.h5")
        next_chords = generate_progression(test_roman, lstm, num_steps=2)
        print(f"Generated next chords: {next_chords}")
    except FileNotFoundError as e:
        print(f"Note: {e}")