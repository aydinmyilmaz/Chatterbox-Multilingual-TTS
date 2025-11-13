# Chatterbox Multilingual TTS - FastAPI Server

FastAPI version of the Chatterbox Multilingual TTS, converted from Gradio for RunPod deployment.

## 🚀 Quick Start

### RunPod Deployment (Recommended)

```bash
# 1. Clone repository
cd /workspace
git clone https://github.com/YOUR_USERNAME/Chatterbox-Multilingual-TTS.git
cd Chatterbox-Multilingual-TTS

# 2. Run setup script
bash scripts/setup_runpod.sh

# 3. Start server
source venv/bin/activate
python server.py
```

### Local Installation

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
├── server.py                 # FastAPI server (main entry point)
├── requirements_fastapi.txt  # FastAPI dependencies
├── requirements.txt          # Original Gradio dependencies
├── scripts/
│   ├── setup_runpod.sh      # Setup script (dependencies installation)
│   ├── RUNPOD_SETUP.sh      # RunPod full setup (clone + setup)
│   └── GITHUB_PUSH.sh       # GitHub push helper script
├── docs/
│   ├── README_FASTAPI.md    # Detailed API documentation
│   ├── DEPLOY_RUNPOD.md     # RunPod deployment guide
│   └── PROJECT_STRUCTURE.md # Project structure documentation
├── examples/
│   └── app.py               # Original Gradio app (for reference)
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

- **[docs/README_FASTAPI.md](docs/README_FASTAPI.md)** - Detailed API documentation
- **[docs/DEPLOY_RUNPOD.md](docs/DEPLOY_RUNPOD.md)** - RunPod deployment guide
- **[docs/PROJECT_STRUCTURE.md](docs/PROJECT_STRUCTURE.md)** - Project structure details

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
