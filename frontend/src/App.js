import React, { useState } from "react";

function App() {
  const [files, setFiles] = useState([]);
  const [chords, setChords] = useState(() => []);

  const handleUpload = async () => {
    if (!files.length) {
      alert("Select at least one file");
      return;
    }

    const formData = new FormData();

    for (let i = 0; i < files.length; i++) {
      formData.append("audio", files[i]);
    }

    try {
      const response = await fetch("http://localhost:5000/analyze", {
        method: "POST",
        body: formData
      });

      const data = await response.json();
      console.log("Backend returned:", data);

      if (data.success && Array.isArray(data.chords)) {
        setChords(data.chords);
      } else {
        console.error("Unexpected response format");
        setChords([]);
      }

    } catch (error) {
      console.error("Error communicating with backend:", error);
      setChords([]);
    }
  };

  console.log("Rendered chords:", chords);

  return (
    <div style={{ padding: "40px" }}>
      <h1>AI Guitar Chord Classifier</h1>

      <input
        type="file"
        accept=".wav"
        multiple
        onChange={(e) => setFiles(e.target.files)}
      />

      <br /><br />

      <button onClick={handleUpload}>
        Analyze
      </button>

      <div style={{ marginTop: "30px" }}>
        <h2>Detected Chords:</h2>
        {Array.isArray(chords) && chords.length > 0 ? (
          <div>
            {chords.map((item, index) => (
              <div key={index} style={{ marginBottom: "20px" }}>
                <strong>{item.filename}</strong>
                <div>Chord: {item.chord}</div>
                <div>Confidence: {(item.confidence * 100).toFixed(2)}%</div>
              </div>
            ))}
          </div>
        ) : (
          <p>No chords detected yet</p>
        )}
      </div>
    </div>
  );
}

export default App;