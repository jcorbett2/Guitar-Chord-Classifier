import os
import subprocess

def convert_m4a_to_wav(root="datasets"):
    for subdir, dirs, files in os.walk(root):

        # Only convert inside actual chord folders (skip root)
        if subdir == root:
            continue

        # Count existing WAV files
        wav_files = [f for f in files if f.lower().endswith(".wav")]
        wav_count = len(wav_files)

        print(f"\nFolder: {subdir}")
        print(f" - WAV files found: {wav_count}")

        # If already 3 or more WAVs → Skip folder
        if wav_count >= 3:
            print(" - Skipping: already has 3 or more WAV files.")
            continue

        # Convert only missing files
        for file in files:
            if file.lower().endswith(".m4a"):
                m4a_path = os.path.join(subdir, file)
                wav_path = os.path.join(subdir, file[:-4] + ".wav")

                print(f"   Converting: {file} → {file[:-4]}.wav")

                result = subprocess.run(
                    ["ffmpeg", "-y", "-i", m4a_path, wav_path],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )

                if result.returncode != 0:
                    print(f"   ERROR converting {file}")
                else:
                    print(f"   ✔ Successfully converted {file}")

        # Recount after conversions
        final_wavs = len([f for f in os.listdir(subdir) if f.endswith(".wav")])
        print(f" - Final WAV count: {final_wavs}")

    print("\nConversion complete.")

convert_m4a_to_wav("datasets")
