# Chatterbox Multilingual TTS - FastAPI Server

FastAPI version of the Chatterbox Multilingual TTS, converted from Gradio for RunPod deployment.

## Features

- ✅ Generate speech from text in 23 languages
- ✅ Upload and use reference audio files for voice cloning
- ✅ RESTful API endpoints
- ✅ RunPod ready

## Installation

```bash
# Install dependencies
pip install -r requirements_fastapi.txt

# Install PyTorch (CUDA version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

## Usage

### Start Server

```bash
python server.py
```

Or with uvicorn:
```bash
uvicorn server:app --host 0.0.0.0 --port 8004
```

### API Endpoints

#### 1. Generate Speech
```bash
POST /generate
Content-Type: application/json

{
  "text": "Hello, this is a test",
  "language_id": "en",
  "audio_prompt_path": "reference_audio/my_voice.wav",  # optional
  "exaggeration": 0.5,
  "temperature": 0.8,
  "seed": 0,
  "cfg_weight": 0.5
}
```

#### 2. Upload Reference Audio
```bash
POST /upload_reference
Content-Type: multipart/form-data

file: <audio_file>
name: "my_voice"  # optional
```

#### 3. List Supported Languages
```bash
GET /languages
```

#### 4. List Uploaded References
```bash
GET /references
```

#### 5. Health Check
```bash
GET /health
```

## Example Usage

### Python
```python
import requests

# Generate speech
response = requests.post(
    "http://localhost:8004/generate",
    json={
        "text": "Geçen ay YouTube kanalımızda iki milyar görüntüleme ile yeni bir dönüm noktasına ulaştık.",
        "language_id": "tr",
        "temperature": 0.8,
        "cfg_weight": 0.5
    }
)

with open("output.wav", "wb") as f:
    f.write(response.content)
```

### cURL
```bash
curl -X POST "http://localhost:8004/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Hello world",
    "language_id": "en"
  }' \
  --output output.wav
```

## Supported Languages

- Arabic (ar)
- Danish (da)
- German (de)
- Greek (el)
- English (en)
- Spanish (es)
- Finnish (fi)
- French (fr)
- Hebrew (he)
- Hindi (hi)
- Italian (it)
- Japanese (ja)
- Korean (ko)
- Malay (ms)
- Dutch (nl)
- Norwegian (no)
- Polish (pl)
- Portuguese (pt)
- Russian (ru)
- Swedish (sv)
- Swahili (sw)
- Turkish (tr)
- Chinese (zh)

## RunPod Deployment

1. Clone this repository
2. Install dependencies
3. Start server:
   ```bash
   python server.py
   ```
4. Server will be available on port 8004 (or PORT environment variable)

## Notes

- Model is loaded at startup
- Reference audio files are stored in `./reference_audio/`
- Maximum text length: 300 characters
- Default voice is used if no reference audio is provided

