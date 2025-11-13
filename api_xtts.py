"""
FastAPI server for Coqui TTS (XTTS v2) model
Separate API endpoint running on different port to avoid dependency conflicts
"""

import os
import io
import sys
import logging
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import soundfile as sf
import torch

# Setup logging with UTF-8 encoding support
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
# Ensure UTF-8 encoding for logging
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

logger = logging.getLogger(__name__)

# Patch torch.load for TTS compatibility
_original_torch_load = torch.load
def patched_load(*args, **kwargs):
    kwargs["weights_only"] = False
    return _original_torch_load(*args, **kwargs)
torch.load = patched_load

# Device configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"🚀 Running on device: {DEVICE}")

# Global TTS model
TTS_MODEL = None

# Reference audio storage
REFERENCE_AUDIO_DIR = Path("./reference_audio_xtts")
REFERENCE_AUDIO_DIR.mkdir(exist_ok=True)

# Supported languages for XTTS v2
SUPPORTED_LANGUAGES = {
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "it": "Italian",
    "pt": "Portuguese",
    "pl": "Polish",
    "tr": "Turkish",
    "ru": "Russian",
    "nl": "Dutch",
    "cs": "Czech",
    "ar": "Arabic",
    "zh-cn": "Chinese",
    "ja": "Japanese",
    "hu": "Hungarian",
    "ko": "Korean",
}

def get_or_load_model():
    """Load or get the TTS model."""
    global TTS_MODEL
    if TTS_MODEL is None:
        logger.info("Loading Coqui TTS (XTTS v2) model...")
        try:
            from TTS.api import TTS
            TTS_MODEL = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=(DEVICE == "cuda"))
            logger.info("✅ Coqui TTS model loaded successfully")
        except ImportError:
            logger.error("❌ TTS package not found. Install with: pip install TTS==0.22.0")
            raise
        except Exception as e:
            logger.error(f"❌ Failed to load TTS model: {e}")
            raise
    return TTS_MODEL

# FastAPI app
app = FastAPI(
    title="Coqui TTS (XTTS v2) API",
    version="1.0.0",
    description="Separate API endpoint for Coqui TTS XTTS v2 model"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request models
class XTTSRequest(BaseModel):
    text: str = Field(..., description="Text to synthesize")
    language: str = Field(..., description="Language code (e.g., en, tr, es, fr)")
    speaker_wav_filename: Optional[str] = Field(None, description="Filename of reference speaker audio (must be uploaded first)")
    file_path: Optional[str] = Field(None, description="Optional output file path (for internal use)")

# Startup event - load model
@app.on_event("startup")
async def startup_event():
    """Load model on startup."""
    try:
        get_or_load_model()
        logger.info("✅ Model loaded on startup")
    except Exception as e:
        logger.error(f"❌ Failed to load model on startup: {e}")

# Health check
@app.get("/health")
async def health():
    """Health check endpoint."""
    model_loaded = TTS_MODEL is not None
    return {
        "status": "healthy" if model_loaded else "model_not_loaded",
        "model_loaded": model_loaded,
        "device": DEVICE,
        "cuda_available": torch.cuda.is_available(),
        "model": "coqui_tts_xtts_v2"
    }

# Supported languages
@app.get("/languages")
async def get_languages():
    """Get list of supported languages."""
    return {
        "supported_languages": SUPPORTED_LANGUAGES,
        "count": len(SUPPORTED_LANGUAGES),
        "model": "coqui_tts_xtts_v2"
    }

# Upload reference speaker audio
@app.post("/upload_speaker")
async def upload_speaker(
    file: UploadFile = File(...),
    name: Optional[str] = Form(None)
):
    """
    Upload reference speaker audio file for voice cloning.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")
    
    # Sanitize filename
    safe_filename = name if name else file.filename
    safe_filename = safe_filename.replace(" ", "_").replace("/", "_").replace("\\", "_")
    
    if not safe_filename.lower().endswith((".wav", ".mp3", ".flac", ".ogg", ".m4a")):
        raise HTTPException(
            status_code=400,
            detail="Invalid file type. Only audio files (.wav, .mp3, .flac, .ogg, .m4a) are allowed."
        )
    
    destination_path = REFERENCE_AUDIO_DIR / safe_filename
    
    try:
        content = await file.read()
        with open(destination_path, "wb") as buffer:
            buffer.write(content)
        
        logger.info(f"Uploaded speaker audio: {safe_filename} ({len(content)} bytes)")
        
        return {
            "message": "Speaker audio uploaded successfully",
            "filename": safe_filename,
            "path": str(destination_path),
            "size": len(content)
        }
    except Exception as e:
        logger.error(f"Error uploading speaker audio: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to upload file: {str(e)}")

# List uploaded speaker files
@app.get("/speaker_files")
async def get_speaker_files():
    """List uploaded speaker audio files."""
    files = []
    for file_path in REFERENCE_AUDIO_DIR.glob("*"):
        if file_path.is_file() and file_path.suffix.lower() in ['.wav', '.mp3', '.flac', '.ogg', '.m4a']:
            files.append(file_path.name)
    return sorted(files)

# Generate TTS
@app.post("/tts")
async def generate_tts(request: XTTSRequest):
    """
    Generate speech using Coqui TTS (XTTS v2).
    """
    try:
        # Validate language
        if request.language.lower() not in SUPPORTED_LANGUAGES:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported language '{request.language}'. Supported: {', '.join(SUPPORTED_LANGUAGES.keys())}"
            )
        
        # Get model
        tts_model = get_or_load_model()
        if tts_model is None:
            raise HTTPException(status_code=503, detail="TTS model is not loaded")
        
        # Resolve speaker audio path
        speaker_wav_path = None
        if request.speaker_wav_filename:
            speaker_path = REFERENCE_AUDIO_DIR / request.speaker_wav_filename
            if not speaker_path.exists():
                raise HTTPException(
                    status_code=404,
                    detail=f"Speaker audio file '{request.speaker_wav_filename}' not found. Please upload it first via /upload_speaker"
                )
            speaker_wav_path = str(speaker_path)
        
        if not speaker_wav_path:
            raise HTTPException(
                status_code=400,
                detail="speaker_wav_filename is required for XTTS v2"
            )
        
        logger.info(f"Generating audio for text: '{request.text[:50]}...' (lang: {request.language})")
        
        # Generate audio using TTS
        # XTTS v2 requires speaker_wav, so we use tts_to_file and read it back
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            tmp_path = tmp_file.name
        
        try:
            tts_model.tts_to_file(
                text=request.text,
                file_path=tmp_path,
                speaker_wav=speaker_wav_path,
                language=request.language.lower()
            )
            
            # Read generated audio
            audio_data, sample_rate = sf.read(tmp_path)
            
            # Convert to mono if stereo
            if len(audio_data.shape) > 1:
                audio_data = audio_data[:, 0]
            
            logger.info(f"Audio generation complete. Sample rate: {sample_rate}Hz, Length: {len(audio_data)} samples")
            
            # Create audio buffer
            audio_buffer = io.BytesIO()
            sf.write(audio_buffer, audio_data, sample_rate, format="WAV")
            audio_buffer.seek(0)
            
            return StreamingResponse(
                audio_buffer,
                media_type="audio/wav",
                headers={
                    "Content-Disposition": f'attachment; filename="xtts_output.wav"',
                    "X-Sample-Rate": str(sample_rate)
                }
            )
        finally:
            # Clean up temp file
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating TTS: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to generate audio: {str(e)}")

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API info."""
    return {
        "name": "Coqui TTS (XTTS v2) API",
        "version": "1.0.0",
        "description": "Separate API endpoint for Coqui TTS XTTS v2 model",
        "endpoints": {
            "POST /tts": "Generate TTS audio",
            "POST /upload_speaker": "Upload reference speaker audio file",
            "GET /speaker_files": "List uploaded speaker audio files",
            "GET /languages": "Get supported languages",
            "GET /health": "Health check"
        },
        "supported_languages_count": len(SUPPORTED_LANGUAGES),
        "port": 8005,
        "note": "This API runs on port 8005 to avoid dependency conflicts with Chatterbox TTS API (port 8004)"
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8005))  # Default port 8005
    uvicorn.run(app, host="0.0.0.0", port=port)

