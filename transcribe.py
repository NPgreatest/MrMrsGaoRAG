import os
import json
import whisper
import torch

# Define input and output directories
audio_dir = "./audios/"
transcribe_dir = "./transcribe/"

# Ensure the transcribe directory exists
os.makedirs(transcribe_dir, exist_ok=True)

# Check if GPU is available and use CUDA
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load the Whisper model (large model recommended for Chinese transcription)
model = whisper.load_model("medium", device=device)  # Use "large" for better Chinese support

stop = 999

# Process each audio file in the folder
for i,filename in enumerate(os.listdir(audio_dir)):
    if filename.lower().endswith(('.mp3', '.wav', '.m4a', '.flac', '.ogg')):
        audio_path = os.path.join(audio_dir, filename)
        json_filename = os.path.splitext(filename)[0] + ".json"
        json_path = os.path.join(transcribe_dir, json_filename)

        # Skip if transcription file already exists
        if os.path.exists(json_path):
            print(f"Skipping {filename}, transcription already exists.")
            continue

        print(f"Transcribing {filename}...")
        result = model.transcribe(audio_path, language="zh")  # Force Chinese transcription

        # Save the transcription to a JSON file
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=4)

        print(f"Saved transcription to {json_path}")

        if i==stop:
            break

print("Transcription process completed.")
