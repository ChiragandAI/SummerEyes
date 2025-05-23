from faster_whisper import WhisperModel
import shutil
import os, io

# Ensure the directory exists to save uploaded files
os.makedirs("temp_audio", exist_ok=True)

# Load the Whisper model (options: 'tiny', 'base', 'small', 'medium', 'large')
model = WhisperModel("base", compute_type="int8")  # Good balance between speed and accuracy

def transcribe(file):
    # Save uploaded file to a temp directory
    file_location = f"temp_audio/{file['filename']}"

    with open(file_location, "wb") as buffer:
        shutil.copyfileobj(io.BytesIO(file['file']), buffer)

    # Transcribe audio using the Whisper model
    segments, _ = model.transcribe(file_location)
    
    # Combine all text segments into one string
    text = " ".join([segment.text for segment in segments])

    return {"text": text}