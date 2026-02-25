from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import uuid

import chord_detector  # your existing file

app = Flask(__name__)
CORS(app)

# 🔥 Load model ONCE at startup (important)
MODEL_PATH = "models/chord_cnn_chroma.keras"
model = chord_detector.load_cnn_model(MODEL_PATH)
class_names = chord_detector.load_class_names()
expected_n, expected_t = chord_detector.get_model_input_shape(model)

UPLOAD_FOLDER = "temp_uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@app.route("/analyze", methods=["POST"])
def analyze():
    if "audio" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    files = request.files.getlist("audio")

    # Clear folder
    for f in os.listdir(UPLOAD_FOLDER):
        os.remove(os.path.join(UPLOAD_FOLDER, f))

    # Save uploads
    for file in files:
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)

    raw_results = chord_detector.detect_chords_from_directory(
        UPLOAD_FOLDER,
        model,
        class_names,
        expected_n,
        expected_t
    )

    # 🔥 Normalize results into clean JSON structure
    formatted_results = []

    for item in raw_results:
        # If it's a tuple like (filename, chord, confidence)
        if isinstance(item, (list, tuple)) and len(item) >= 3:
            formatted_results.append({
                "filename": item[0],
                "chord": item[1],
                "confidence": float(item[2])
            })

        # If it's already a dict, just append it
        elif isinstance(item, dict):
            formatted_results.append(item)

    # Cleanup
    for f in os.listdir(UPLOAD_FOLDER):
        os.remove(os.path.join(UPLOAD_FOLDER, f))


    print("FORMATTED RESULTS:", formatted_results)
    return jsonify({
        "success": True,
        "chords": formatted_results
    })
    


if __name__ == "__main__":
    app.run(port=5000, debug=True)