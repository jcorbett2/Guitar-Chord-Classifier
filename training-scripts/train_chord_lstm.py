import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.sequence import pad_sequences
import json


# ============================================================
# 1. BUILT-IN CHORD PROGRESSION DATASET (Roman Numeral Based)
# ============================================================

# These are *real harmonic progressions* – very common across genres
PROGRESSIONS = [
    # Pop / Rock
    "I V vi IV",
    "vi IV I V",
    "I vi IV V",
    "I IV V I",
    "I V I V",
    "I V IV I",

    # ii-V-I Jazz / Pop Cadence
    "ii V I",
    "ii V I vi",
    "ii V I IV",

    # Minor key patterns (natural minor)
    "i VII VI VII",
    "i VI III VII",
    "i iv v i",
    "i VI VII i",

    # Blues-ish / modal-ish
    "I bVII IV I",
    "I IV bVII I",

    # Circle of Fifths
    "I IV vii° iii vi ii V I",
    "I ii V I",

    # Standard cadences
    "IV V I",
    "V IV I",
    "vi ii V I",

    # Variations
    "iii vi ii V",
    "I iii vi IV V I",
    "I ii iii IV V I"
]


# ============================================================
# 2. TOKENIZATION / ENCODING
# ============================================================

def build_vocab(progressions):
    """Create a mapping of chord → integer ID"""
    tokens = set()

    for line in progressions:
        for chord in line.split():
            tokens.add(chord)

    # Reserve 0 for padding
    vocab = {tok: i+1 for i, tok in enumerate(sorted(tokens))}
    inv_vocab = {i: tok for tok, i in vocab.items()}

    return vocab, inv_vocab


VOCAB, INV_VOCAB = build_vocab(PROGRESSIONS)
VOCAB_SIZE = len(VOCAB) + 1  # +1 for padding ID=0


def encode(chords):
    return [VOCAB[c] for c in chords]


def decode(id_):
    return INV_VOCAB.get(id_, "PAD")


# ============================================================
# 3. BUILD TRAINING SEQUENCES
# ============================================================

def build_training_data(sequences, seq_len=4):
    X = []
    y = []

    for line in sequences:
        chords = line.split()
        encoded = encode(chords)

        # Generate sliding-window sequences
        for i in range(1, len(encoded)):
            input_seq = encoded[:i]
            target = encoded[i]

            X.append(input_seq)
            y.append(target)

    # Pad inputs to equal length
    X = pad_sequences(X, maxlen=seq_len, padding='pre')

    return np.array(X), np.array(y)


SEQ_LEN = 8
X, y = build_training_data(PROGRESSIONS, seq_len=SEQ_LEN)


# ============================================================
# 4. BUILD THE LSTM MODEL
# ============================================================

def build_model(vocab_size, embedding_dim=32, lstm_units=64):
    model = Sequential([
        Embedding(vocab_size, embedding_dim, input_length=SEQ_LEN),
        LSTM(lstm_units, return_sequences=False),
        Dense(vocab_size, activation='softmax')
    ])

    model.compile(loss="sparse_categorical_crossentropy",
                  optimizer="adam",
                  metrics=["accuracy"])
    return model


model = build_model(VOCAB_SIZE)


# ============================================================
# 5. TRAIN MODEL
# ============================================================

print("Training model...")
history = model.fit(
    X, y,
    epochs=80,
    batch_size=16,
    verbose=1
)

model.save("chord_lstm.h5")
print("\nModel saved as chord_lstm.h5")

# Save vocab too
with open("vocab.json", "w") as f:
    json.dump(VOCAB, f)
print("Saved vocab.json")


# ============================================================
# 6. GENERATION FUNCTION
# ============================================================

def predict_next(model, seq):
    encoded = encode(seq)
    padded = pad_sequences([encoded], maxlen=SEQ_LEN, padding='pre')

    probs = model.predict(padded)[0]
    next_id = np.argmax(probs)
    return decode(next_id)


def generate_progression(seed, length=8):
    """Generate a chord progression starting from a seed sequence"""
    seq = seed[:]

    for _ in range(length):
        next_chord = predict_next(model, seq)
        seq.append(next_chord)

    return seq


# ============================================================
# 7. TEST GENERATION
# ============================================================

print("\n=== TEST GENERATION ===")
seed = ["I", "V"]
generated = generate_progression(seed, length=6)
print("Seed:", seed)
print("Generated progression:", " ".join(generated))
