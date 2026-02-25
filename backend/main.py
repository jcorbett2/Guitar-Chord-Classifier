import sys
import os
from chord_detector import load_cnn_model, load_class_names, get_model_input_shape, detect_chords_from_directory
from key_analyzer import find_most_likely_key, extract_roman_numerals, get_roman_numeral, ROMAN_NUMERALS, KEYS
from progression_generator import load_lstm_model, generate_progression



## Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import warnings
warnings.filterwarnings('ignore')
try:
    from absl import logging as absl_logging
    absl_logging.set_verbosity(absl_logging.ERROR)
except Exception:
    pass
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)



def roman_to_chord(roman_numeral, key):
    # Convert Roman numeral to actual chord in given key
    diatonic = KEYS.get(key, [])
    
    # Reverse mapping: Roman numeral position to scale degree
    roman_to_degree = {
        "I": 0, "ii": 1, "iii": 2, "IV": 3, "V": 4, "vi": 5, "vii°": 6
    }
    
    degree_idx = roman_to_degree.get(roman_numeral)
    if degree_idx is None or degree_idx >= len(diatonic):
        return "?"
    
    return diatonic[degree_idx]

def format_progression_with_chords(roman_sequence, key):
    pairs = []
    for rn in roman_sequence:
        chord = roman_to_chord(rn, key)
        pairs.append(f"{rn} ({chord})")
    
    return " → ".join(pairs)

def main():
    # Parse command line arguments
    if len(sys.argv) < 2:
        print("=" * 50)
        print("CHORD DETECTION & PROGRESSION GENERATOR")
        print("=" * 50)
        print("\nHow to use: python main.py <directory> [--cnn-model <path>] [--lstm-model <path>]")
        print("\nExample:")
        print("  python main.py ./user_input/test1")
        print("  python main.py ./user_input/test1 --cnn-model models/chord_cnn_chroma.keras")
        sys.exit(1)
    
    audio_dir = sys.argv[1]

    ## Load trained models
    cnn_model_path = "models/chord_cnn_chroma.keras"
    lstm_model_path = "models/chord_lstm.h5"
    
    ## Parse optional model paths
    for i, arg in enumerate(sys.argv[2:]):
        if arg == "--cnn-model" and i + 3 < len(sys.argv):
            cnn_model_path = sys.argv[i + 3]
        elif arg == "--lstm-model" and i + 3 < len(sys.argv):
            lstm_model_path = sys.argv[i + 3]
    
    ## If no directory exists print error
    if not os.path.isdir(audio_dir):
        print(f"Directory not found: {audio_dir}")
        sys.exit(1)
    
    print("\n" + "=" * 50)
    print("CHORD DETECTION & PROGRESSION GENERATOR")
    print("=" * 50 + "\n")
    



    ## start chord detection
    print("Step 1: Loading CNN model...")
    try:
        cnn_model = load_cnn_model(cnn_model_path)
        class_names = load_class_names()
        expected_n, expected_t = get_model_input_shape(cnn_model)
        print(f"CNN loaded (expects input: {expected_n}x{expected_t})")
    except Exception as e:
        print(f"Failed to load CNN model: {e}")
        sys.exit(1)
    



    ## Detect chords from audio files
    print(f"\nStep 2: Detecting chords from '{audio_dir}'...\n")
    results = detect_chords_from_directory(audio_dir, cnn_model, class_names, expected_n, expected_t)
    
    if not results: ## Throw error if no chords detected
        print("No chords detected.")
        sys.exit(1)
    
    detected_chords = [chord for _, chord, _ in results]
    print(f"\nDetected {len(detected_chords)} chord(s): {detected_chords}\n")
    


    ## Analyze key of detected chords
    print("Step 3: Analyzing key...\n")
    key, diatonic_chords, mapped_chords = find_most_likely_key(detected_chords)
    
    print(f"Most likely key: {key}")
    print(f"Diatonic chords: {diatonic_chords}\n")
    
    print("Roman numeral mapping:")
    for chord, degree in mapped_chords:
        if degree is None:
            print(f"  {chord}: (non-diatonic)")
        else:
            rn = get_roman_numeral(degree)
            print(f"  {chord}: {rn}")
    
    ## Extract Roman numeral sequence
    roman_sequence = extract_roman_numerals(mapped_chords)
    print(f"\nRoman numeral sequence: {roman_sequence}\n")
    
    


    ## Prompt for progression generation
    print("-" * 50)
    user_input = input("Generate chord progression? (y/n): ").strip().lower()
    
    if user_input != 'y':
        print("Exiting.")
        sys.exit(0)
    



    ## Load LSTM model and generate progression
    print("\nStep 4: Loading LSTM model...\n")
    try:
        lstm_model = load_lstm_model(lstm_model_path)
        print(f"LSTM loaded")
    except FileNotFoundError as e:
        print(f"Warning: {e}")
        print("Skipping progression generation.\n")
        sys.exit(0)
    except Exception as e:
        print(f"Failed to load LSTM: {e}")
        sys.exit(1)
    




    ## Generate next chords
    print("Step 5: Generating progression...\n")
    
    num_steps = 2  # can parameterize this
    generated = generate_progression(roman_sequence, lstm_model, num_steps=num_steps)
    
    # Combine detected + generated progressions
    full_progression = roman_sequence + generated
    
    print("=" * 50)
    print("RESULTS")
    print("=" * 50)
    print(f"\nKey: {key}\n")
    
    print("Detected progression (with chords):")
    print(f"  {format_progression_with_chords(roman_sequence, key)}\n")

    print("Generated next chord(s) (with chords):")
    print(f"  {format_progression_with_chords(generated, key)}\n")
    
    print("=" * 50)
    print("Full Progression:")
    print("=" * 50)
    
    print(f"With chords in {key}:")
    print(f"  {format_progression_with_chords(full_progression, key)}\n")




if __name__ == "__main__":
    main()