from collections import defaultdict
import re

# Musical key definitions
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
    "A Minor": ["Am", "Bdim", "C", "Dm", "Em", "F", "G"],
    "E Minor": ["Em", "F#dim", "G", "Am", "Bm", "C", "D"],
    "B Minor": ["Bm", "C#dim", "D", "Em", "F#m", "G", "A"],
    "F# Minor": ["F#m", "G#dim", "A", "Bm", "C#m", "D", "E"],
    "C# Minor": ["C#m", "D#dim", "E", "F#m", "G#m", "A", "B"],
    "G# Minor": ["G#m", "A#dim", "B", "C#m", "D#m", "E", "F#"],
    "D# Minor": ["D#m", "E#dim", "F#", "G#m", "A#m", "B", "C#"],
    "A# Minor": ["A#m", "B#dim", "C#", "D#m", "E#m", "F#", "G#"],
}

## Roman numeral mappings
ROMAN_NUMERALS = {
    1: "I",
    2: "ii",
    3: "iii",
    4: "IV",
    5: "V",
    6: "vi",
    7: "vii°"
}




## Normalize chord names from model output to standard notation
## A major -> A, A minor -> Am, etc.
def normalize_chord_name(chord_name):

    chord_name = chord_name.strip()
    
    # Remove common suffixes
    chord_name = re.sub(r'(?i)(major|maj|minor|min)', lambda m: 'm' if m.group(1).lower().startswith('min') else '', chord_name)
    chord_name = chord_name.strip()
    
    # If ends with 'm', keep it; if no suffix, assume major (no suffix)
    # Already handled above
    
    return chord_name





## Determine the most likely key from a list of detected chords
def find_most_likely_key(predicted_chords):

    # Normalize all chord names
    normalized_chords = [normalize_chord_name(c) for c in predicted_chords]
    
    key_scores = defaultdict(int)
    
    for chord in normalized_chords:
        root = chord[:-1] if chord[-1] in "mM" else chord
        for key, diatonic in KEYS.items():
            if chord in diatonic:
                key_scores[key] += 2
            elif root in [d[:-1] if d[-1] in "mM" else d for d in diatonic]:
                key_scores[key] += 1
    
    # Handle case where no keys score (fallback to A Major)
    if not key_scores:
        print("Warning: No matching keys found. Defaulting to A Major.")
        best_key = "A Major"
    else:
        ranked = sorted(key_scores.items(), key=lambda x: x[1], reverse=True)
        best_score = ranked[0][1]
        likely_keys = [k for k, s in ranked if s == best_score]
        best_key = likely_keys[0]
    
    diatonic_chords = KEYS[best_key]
    
    # Map each chord to its degree (1-7) or None if non-diatonic
    mapped = []
    for chord in normalized_chords:
        if chord in diatonic_chords:
            degree = diatonic_chords.index(chord) + 1
            mapped.append((chord, degree))
        else:
            mapped.append((chord, None))
    
    return best_key, diatonic_chords, mapped






## Convert scale degree to Roman numeral then extract sequence
def get_roman_numeral(degree):
    return ROMAN_NUMERALS.get(degree, "?")

def extract_roman_numerals(mapped_chords):
    roman_seq = []
    for chord, degree in mapped_chords:
        if degree is not None:
            roman_seq.append(get_roman_numeral(degree))
    return roman_seq






if __name__ == "__main__":
    # Quick test
    test_chords = ["Cmajor", "Gmajor", "Aminor", "Fmajor"]
    normalized = [normalize_chord_name(c) for c in test_chords]
    print(f"Original: {test_chords}")
    print(f"Normalized: {normalized}")
    
    key, diatonic, mapped = find_most_likely_key(test_chords)
    print(f"Key: {key}")
    print(f"Diatonic chords: {diatonic}")
    print(f"Mapped chords: {mapped}")
    
    roman = extract_roman_numerals(mapped)
    print(f"Roman numerals: {roman}")