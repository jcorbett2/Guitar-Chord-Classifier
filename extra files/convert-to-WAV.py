import os
import subprocess

##by changing this directory it will convert all m4a files in it to wav files
directory = "user_input/test6"

def convert_m4a_to_wav_and_delete(root=directory):
    for subdir, dirs, files in os.walk(root):
        for file in files:
            if file.lower().endswith(".m4a"):
                m4a_path = os.path.join(subdir, file)
                wav_path = os.path.join(subdir, file[:-4] + ".wav")

                # Skip if .wav already exists
                if os.path.exists(wav_path):
                    print(f"Skipping (WAV exists): {m4a_path}")
                    continue

                print(f"Converting: {m4a_path} → {wav_path}")

                # Run FFmpeg conversion
                result = subprocess.run(
                    [
                        "ffmpeg", "-y",
                        "-i", m4a_path,
                        wav_path
                    ],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )

                # Check success
                if result.returncode == 0:
                    print(f"Conversion successful. Deleting {m4a_path}")
                    os.remove(m4a_path)
                else:
                    print(f"❌ Conversion failed for {m4a_path}")
                    print(result.stderr.decode())


if __name__ == "__main__":
    convert_m4a_to_wav_and_delete("user_input/test6")
