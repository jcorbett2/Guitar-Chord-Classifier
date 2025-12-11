import numpy as np
from collections import defaultdict
import tensorflow as tf
import os

# -------------------------------
#        KEY DICTIONARY
# -------------------------------
KEYS = {
    "C Major": ["C", "Dm", "Em", "F", "G", "Am", "Bdim"],
    "G Major": ["G", "Am", "Bm", "C", "D", "Em", "F#dim"],
    "D Major": ["D", "Em", "F#m", "G", "A", "Bm", "C#dim"],
    "A Major": ["A", "Bm", "C#m", "D", "E", "F#m", "G#dim"],
    "E Major": ["E", "F#m", "G#m", "A", "B", "C#m", "D#dim"],
    "B Major": ["B", "C#m", "D#m", "E", "F#", "G#m", "A#dim"],
    "F# Major": ["F#", "G#m", "A#m", "B", "C#", "D#m", "E#dim"],
    "C# Major": ["C#", "D#m", "E#m", "F#", "G#", "A#m", "B#dim"],

    "F Major": ["F", "Gm", "Am", "Bb", "C", "Dm", "Edim"],
    "Bb Major": ["Bb", "Cm", "Dm", "Eb", "F", "Gm", "Adim"],
    "Eb Major": ["Eb", "Fm", "Gm", "Ab", "Bb", "Cm", "Ddim"],
    "Ab Major": ["Ab", "Bbm", "Cm", "Db", "Eb", "Fm", "Gdim"],
    "Db Major": ["Db", "Ebm", "Fm", "Gb", "Ab", "Bbm", "Cdim"],
    "Gb Major": ["Gb", "Abm", "Bbm", "Cb", "Db", "Ebm", "Fdim"],
    "Cb Major": ["Cb", "Dbm", "Ebm", "Fb", "Gb", "Abm", "Bbdim"],

    # Minor keys
    "A Minor": ["Am", "Bdim", "C", "Dm", "Em", "F", "G"],
    "E Minor": ["Em", "F#dim", "G", "Am", "Bm", "C", "D"],
    "B Minor": ["Bm", "C#dim", "D", "Em", "F#m", "G", "A"],
    "F# Minor": ["F#m", "G#dim", "A", "Bm", "C#m", "D", "E"],
    "C# Minor": ["C#m", "D#dim", "E", "F#m", "G#m", "A", "B"],
    "G# Minor": ["G#m", "A#dim", "B", "C#m", "D#m", "E", "F#"],
    "D# Minor": ["D#m", "E#dim", "F#", "G#m", "A#m", "B", "C#"],
    "A# Minor": ["A#m", "B#dim", "C#", "D#m", "E#m", "F#", "G#"],
}

# Roman numeral mapping for degrees
ROMAN = {
    1: "I",
    2: "ii",
    3: "iii",
    4: "IV",
    5: "V",
    6: "vi",
    7: "vii°"
}

# ------------------------------------
#        FIND MOST LIKELY KEY
# ------------------------------------
def find_most_likely_key(predicted_chords):
    key_scores = defaultdict(int)

    for chord in predicted_chords:
        root = chord[:-1] if chord[-1] in "mM" else chord
        for key, diatonic in KEYS.items():
            if chord in diatonic:
                key_scores[key] += 2
            elif root in [d[:-1] for d in diatonic]:
                key_scores[key] += 1

    ranked = sorted(key_scores.items(), key=lambda x: x[1], reverse=True)
    best_score = ranked[0][1]
    likely_keys = [k for k, s in ranked if s == best_score]
    best_key = likely_keys[0]

    diatonic_chords = KEYS[best_key]

    mapped = []
    for chord in predicted_chords:
        if chord in diatonic_chords:
            degree = diatonic_chords.index(chord) + 1
            mapped.append((chord, degree))
        else:
            mapped.append((chord, None))

    return best_key, diatonic_chords, mapped


# ------------------------------------
#          LSTM LOADING / INFERENCE
# ------------------------------------
def load_lstm_model():
    model_path = os.path.join("models", "chord_lstm.h5")
    return tf.keras.models.load_model(model_path)


def encode_roman_numerals(rns):
    mapping = {
        "I": 0, "ii": 1, "iii": 2, "IV": 3, "V": 4, "vi": 5, "vii°": 6
    }
    return [mapping[r] for r in rns if r in mapping]


def decode_prediction(index):
    mapping = ["I", "ii", "iii", "IV", "V", "vi", "vii°"]
    return mapping[index]


def suggest_progression(roman_sequence, lstm_model):
    encoded = encode_roman_numerals(roman_sequence)
    X = np.array(encoded)[np.newaxis, :, np.newaxis]  # shape (1, seq_len, 1)

    preds = lstm_model.predict(X, verbose=0)
    next_idx = np.argmax(preds)
    return decode_prediction(next_idx)


# ------------------------------------
#        MAIN PIPELINE FUNCTION
# ------------------------------------
def analyze_chords_and_suggest_progression(chords):
    print("\n============================")
    print(" 🎵 CHORD & KEY ANALYSIS")
    print("============================\n")

    print("Detected chords:", chords, "\n")

    # --- Key detection ---
    key, diatonic, mapped = find_most_likely_key(chords)

    print(f"🎼 Most likely key: **{key}**")
    print(f"Diatonic chords: {diatonic}\n")

    # --- Roman numeral translation ---
    roman_seq = []
    print("Roman numeral mapping:")
    for chord, degree in mapped:
        if degree is None:
            print(f"  {chord}: (non-diatonic)")
        else:
            rn = ROMAN[degree]
            roman_seq.append(rn)
            print(f"  {chord}: {rn}")

    print("\nRoman numeral input sequence:", roman_seq)

    # --- LSTM prediction ---
    lstm_model = load_lstm_model()
    next_chord = suggest_progression(roman_seq, lstm_model)

    print("\n============================")
    print(" LSTM CHORD SUGGESTION")
    print("============================")
    print(f"Suggested next chord (roman numeral): **{next_chord}**")

    print("\nFinal output:")
    print(f"Key: {key}")
    print(f"Predicted chord progression continuation: {next_chord}")

    roman_seq.append(next_chord)
    print(f"Final Chord Progression: {roman_seq}")

    return {
        "key": key,
        "roman_input": roman_seq,
        "suggested_next": next_chord
    }


# ------------------------------------
# Example usage (ONLY IF RUN DIRECTLY)
# ------------------------------------
if __name__ == "__main__":
    test_chords = ["C", "G", "Am", "F"]
    analyze_chords_and_suggest_progression(test_chords)
