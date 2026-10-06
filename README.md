# Audio Transcription Service

FastAPI service for converting audio files to text using Whisper. Supports direct file uploads, direct audio URLs (`fileURL`), standalone CLI execution, health monitoring, and interactive OpenAPI documentation.

---

## Features

- **Dual Input Modes**:
  - **Direct File Upload**: Upload local audio (`.mp3`, `.wav`, `.m4a`, `.ogg`, `.flac`, `.webm`, etc.) via `file`.
  - **Audio URL Download**: Provide a remote audio link via `fileURL`—the server downloads and transcribes it directly.
- **Standalone Python Script**: `transcribe.py` for direct command-line transcription of local files or URLs.
- **Fast & Efficient**: Powered by `faster-whisper` (CTranslate2) with fallback to `openai-whisper`.
- **Timestamped Segments**: Optional segment timestamps (`start`, `end`, `text`).
- **Language Detection**: Automatic language detection with confidence score.
- **Health Check Endpoints**: `/health` (and probe alias `/healthz`).
- **Interactive OpenAPI Docs**: Available at `/docs` (Swagger UI) and `/redoc`.

---

## Setup & Installation

### 1. Install Dependencies

```bash
# Activate your virtual environment if you have one
# e.g., .venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

> **Note**: `faster-whisper` bundles PyAV decoders, so audio files like MP3, WAV, and M4A work out-of-the-box on Windows without needing external FFmpeg installations.

---

## Usage

### Option 1: Standalone Python Script (`transcribe.py`)

You can transcribe any local audio file or audio URL directly from the terminal:

```bash
# Transcribe a local file
python transcribe.py audio.mp3

# Transcribe directly from a remote audio URL
python transcribe.py https://example.com/recordings/audio.mp3

# Use a specific model size (tiny, base, small, medium, large-v3)
python transcribe.py audio.mp3 --model medium

# Specify audio language (e.g., English)
python transcribe.py audio.mp3 --language en

# Save transcription output to a file and display timestamped segments
python transcribe.py audio.mp3 --output transcript.txt --show-segments
```

---

### Option 2: FastAPI Web Service (`POST /transcribe`)

#### 1. Start the server
```bash
python main.py
```
Or with Uvicorn:
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

#### 2. Transcribe using a Remote URL (`fileURL`)

**Using `curl`:**
```bash
curl -X POST "http://localhost:8000/transcribe" \
  -F "fileURL=https://example.com/audio/sample.mp3" \
  -F "model=medium" \
  -F "include_segments=true"
```

**Using Python `requests` (Form / Multipart):**
```python
import requests

url = "http://localhost:8000/transcribe"
payload = {
    "fileURL": "https://example.com/audio/sample.mp3",
    "model": "medium",
    "include_segments": "false"
}

response = requests.post(url, data=payload)
result = response.json()
print("Transcribed Text:", result["text"])
```

**Using Python `requests` (JSON body):**
```python
import requests

url = "http://localhost:8000/transcribe"
payload = {
    "fileURL": "https://example.com/audio/sample.mp3",
    "model": "medium"
}

response = requests.post(url, json=payload)
print(response.json())
```

---

#### 3. Transcribe using Direct File Upload (`file`)

**Using `curl`:**
```bash
curl -X POST "http://localhost:8000/transcribe" \
  -F "file=@sample.mp3" \
  -F "model=medium"
```

**Using Python `requests`:**
```python
import requests

url = "http://localhost:8000/transcribe"
with open("sample.mp3", "rb") as f:
    response = requests.post(
        url,
        files={"file": ("sample.mp3", f, "audio/mpeg")},
        data={"model": "medium"}
    )

print(response.json())
```

---

#### Sample API Response:
```json
{
  "success": true,
  "filename": "sample.mp3",
  "source": "url",
  "text": "Welcome to the meeting. Today we are going to discuss the project roadmap.",
  "language": "en",
  "language_probability": 0.9982,
  "duration": 4.12,
  "processing_time_seconds": 1.15,
  "model": "medium",
  "engine": "faster-whisper",
  "segments": null
}
```

---

## API Endpoints Overview

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/transcribe` | Convert uploaded audio file or `fileURL` to text |
| `GET` | `/` | Root service info & endpoints |
| `GET` | `/health` | Health check & uptime |
| `GET` | `/healthz` | Container probe alias |
| `GET` | `/docs` | Interactive Swagger UI |
| `GET` | `/redoc` | ReDoc documentation |
