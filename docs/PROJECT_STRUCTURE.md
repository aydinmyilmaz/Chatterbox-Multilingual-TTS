# Project Structure

```
Chatterbox-Multilingual-TTS/
├── server.py                    # FastAPI server (main entry point)
├── app.py                       # Original Gradio app (for reference)
├── requirements.txt             # Original Gradio dependencies
├── requirements_fastapi.txt     # FastAPI dependencies
├── README.md                    # Main documentation
├── .gitignore                  # Git ignore rules
│
├── scripts/                    # Setup and utility scripts
│   └── setup_runpod.sh        # RunPod deployment setup
│
├── docs/                       # Documentation
│   └── README_FASTAPI.md      # Detailed API documentation
│
├── src/                        # Source code
│   └── chatterbox/            # Core TTS implementation
│       ├── __init__.py
│       ├── mtl_tts.py         # Multilingual TTS class
│       ├── tts.py             # Standard TTS class
│       ├── vc.py              # Voice Conversion class
│       └── models/            # Model components
│           ├── s3gen/        # S3Gen model
│           ├── s3tokenizer/   # S3 tokenizer
│           ├── t3/           # T3 model
│           ├── tokenizers/   # Text tokenizers
│           └── voice_encoder/ # Voice encoder
│
└── reference_audio/           # Uploaded reference audio files
    └── .gitkeep              # Keep directory in git
```

## File Descriptions

### Root Files
- **server.py**: FastAPI server implementation
- **app.py**: Original Gradio interface (kept for reference)
- **requirements_fastapi.txt**: Python dependencies for FastAPI version
- **requirements.txt**: Original Gradio dependencies

### Scripts
- **scripts/setup_runpod.sh**: Automated setup script for RunPod deployment

### Documentation
- **README.md**: Main project documentation
- **docs/README_FASTAPI.md**: Detailed API documentation

### Source Code
- **src/chatterbox/**: Core TTS implementation from Hugging Face Space
  - **mtl_tts.py**: Multilingual TTS implementation
  - **models/**: All model components (T3, S3Gen, tokenizers, etc.)

### Data
- **reference_audio/**: Directory for uploaded reference audio files
