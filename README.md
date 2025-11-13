# Chatterbox Multilingual TTS - FastAPI Server

FastAPI version of the Chatterbox Multilingual TTS, converted from Gradio for RunPod deployment.

## 🚀 Quick Start

### Installation

```bash
# Run setup script
bash scripts/setup_runpod.sh
```

### Start Server

```bash
source venv/bin/activate
python server.py
```

Server will be available at `http://localhost:8000`

## 📁 Project Structure

```
.
├── server.py                 # FastAPI server (main file)
├── app.py                    # Original Gradio app
├── requirements.txt          # Original Gradio requirements
├── requirements_fastapi.txt  # FastAPI requirements
├── scripts/
│   └── setup_runpod.sh      # RunPod setup script
├── docs/
│   └── README_FASTAPI.md    # Detailed API documentation
└── src/
    └── chatterbox/          # Core TTS implementation
```

## 🔌 API Endpoints

### Generate Speech
```bash
POST /generate
Content-Type: application/json

{
  "text": "Hello world",
  "language_id": "en",
  "audio_prompt_path": "reference_audio/my_voice.wav",  # optional
  "exaggeration": 0.5,
  "temperature": 0.8,
  "seed": 0,
  "cfg_weight": 0.5
}
```

### Upload Reference Audio
```bash
POST /upload_reference
Content-Type: multipart/form-data

file: <audio_file>
name: "my_voice"  # optional
```

### Other Endpoints
- `GET /languages` - List supported languages
- `GET /references` - List uploaded reference files
- `GET /health` - Health check
- `GET /` - API information

## 🌍 Supported Languages

23 languages: Arabic, Danish, German, Greek, English, Spanish, Finnish, French, Hebrew, Hindi, Italian, Japanese, Korean, Malay, Dutch, Norwegian, Polish, Portuguese, Russian, Swedish, Swahili, Turkish, Chinese

## 📖 Documentation

For detailed API documentation, see [docs/README_FASTAPI.md](docs/README_FASTAPI.md)

## 🐳 RunPod Deployment

1. Clone repository
2. Run setup: `bash scripts/setup_runpod.sh`
3. Start server: `python server.py`
4. Server runs on port 8000 (or PORT env variable)

## 📝 Notes

- Model loads automatically at startup
- Reference audio files stored in `./reference_audio/`
- Maximum text length: 300 characters
- Default voice used if no reference audio provided
