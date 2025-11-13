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

# Reference audio storage (same structure as Chatterbox TTS)
REFERENCE_AUDIO_DIR = Path("./reference_audio")
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

def resolve_audio_prompt(reference_filename: Optional[str]) -> Optional[str]:
    """Resolve reference audio filename to full path (same as Chatterbox TTS)."""
    if reference_filename and str(reference_filename).strip():
        filename = Path(reference_filename).name
        ref_path = REFERENCE_AUDIO_DIR / filename
        if ref_path.exists() and ref_path.is_file():
            return str(ref_path)
        else:
            logger.warning(f"Reference audio file not found: {filename}")
            return None
    return None

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

# Request models (same structure as Chatterbox TTS for compatibility)
class TTSRequest(BaseModel):
    text: str = Field(..., description="Text to synthesize (will be chunked if longer than 300 chars)")
    language_id: str = Field(..., description="Language code (e.g., en, tr, ar, fr)")
    reference_audio_filename: Optional[str] = Field(None, description="Filename of reference audio file (must be uploaded first via /upload_reference)")
    exaggeration: float = Field(0.5, ge=0.25, le=2.0, description="Speech expressiveness (0.25-2.0)")
    temperature: float = Field(0.8, ge=0.05, le=5.0, description="Randomness in generation (0.05-5.0)")
    seed: int = Field(0, description="Random seed (0 for random)")
    cfg_weight: float = Field(0.5, ge=0.0, le=1.0, description="CFG/Pace weight (0.2-1.0)")
    chunk_size: Optional[int] = Field(300, ge=50, le=500, description="Chunk size for long texts (default: 300)")

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

# Upload reference audio (same endpoint as Chatterbox TTS)
@app.post("/upload_reference")
async def upload_reference(files: list[UploadFile] = File(...)):
    """
    Upload reference audio file(s) for voice cloning.
    Maximum 50 files can be uploaded at once.
    Files are stored in the reference_audio directory.
    Same endpoint structure as Chatterbox TTS for client compatibility.
    """
    if len(files) > 50:
        raise HTTPException(
            status_code=400,
            detail=f"Maximum 50 files allowed per upload. Received {len(files)} files."
        )

    uploaded_files = []
    upload_errors = []

    for file in files:
        if not file.filename:
            upload_errors.append({"filename": "Unknown", "error": "File received with no filename."})
            continue

        # Sanitize filename
        safe_filename = file.filename.replace(" ", "_").replace("/", "_").replace("\\", "_")
        destination_path = REFERENCE_AUDIO_DIR / safe_filename

        try:
            # Validate file type
            if not (
                safe_filename.lower().endswith((".wav", ".mp3", ".flac", ".ogg", ".m4a"))
            ):
                raise ValueError("Invalid file type. Only audio files (.wav, .mp3, .flac, .ogg, .m4a) are allowed.")

            content = await file.read()
            with open(destination_path, "wb") as buffer:
                buffer.write(content)

            logger.info(f"Uploaded reference audio: {safe_filename} ({len(content)} bytes)")
            uploaded_files.append({
                "filename": safe_filename,
                "path": str(destination_path),
                "size": len(content)
            })
        except Exception as e_upload:
            error_msg = f"Error processing file '{file.filename}': {str(e_upload)}"
            logger.error(error_msg, exc_info=True)
            upload_errors.append({"filename": file.filename, "error": str(e_upload)})
        finally:
            await file.close()

    # Get all current reference files
    all_current_reference_files = []
    for file_path in REFERENCE_AUDIO_DIR.glob("*"):
        if file_path.is_file() and file_path.suffix.lower() in ['.wav', '.mp3', '.flac', '.ogg', '.m4a']:
            all_current_reference_files.append(file_path.name)
    all_current_reference_files = sorted(all_current_reference_files)

    response_data = {
        "message": f"Processed {len(files)} file(s).",
        "uploaded_files": [f["filename"] for f in uploaded_files],
        "all_reference_files": all_current_reference_files,
        "errors": upload_errors,
    }
    status_code = 200 if not upload_errors or len(uploaded_files) > 0 else 400
    return JSONResponse(content=response_data, status_code=status_code)

# List reference files (same endpoints as Chatterbox TTS)
@app.get("/reference_files")
async def get_reference_files():
    """List uploaded reference audio files."""
    files = []
    for file_path in REFERENCE_AUDIO_DIR.glob("*"):
        if file_path.is_file() and file_path.suffix.lower() in ['.wav', '.mp3', '.flac', '.ogg', '.m4a']:
            files.append(file_path.name)
    return sorted(files)

@app.get("/get_reference_files")
async def get_reference_files_api():
    """Backward compatibility endpoint (same as Chatterbox TTS)."""
    logger.debug("Request for /get_reference_files.")
    try:
        files = []
        for file_path in REFERENCE_AUDIO_DIR.glob("*"):
            if file_path.is_file() and file_path.suffix.lower() in ['.wav', '.mp3', '.flac', '.ogg', '.m4a']:
                files.append(file_path.name)
        return sorted(files)
    except Exception as e:
        logger.error(f"Error getting reference files for API: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail="Failed to retrieve reference audio files."
        )

@app.get("/get_predefined_voices")
async def get_predefined_voices_api():
    """Backward compatibility endpoint (same as Chatterbox TTS)."""
    logger.debug("Request for /get_predefined_voices.")
    try:
        voices = []
        for file_path in REFERENCE_AUDIO_DIR.glob("*"):
            if file_path.is_file() and file_path.suffix.lower() in ['.wav', '.mp3', '.flac', '.ogg', '.m4a']:
                voices.append({
                    "id": file_path.name,
                    "filename": file_path.name,
                    "display_name": file_path.stem.replace("_", " ").title()
                })
        return sorted(voices, key=lambda x: x["filename"])
    except Exception as e:
        logger.error(f"Error getting predefined voices for API: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail="Failed to retrieve predefined voices list."
        )

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
            "POST /tts": "Generate TTS audio (main endpoint)",
            "POST /generate": "Generate TTS audio (alias for /tts)",
            "POST /upload_reference": "Upload reference audio file(s)",
            "GET /reference_files": "List uploaded reference audio files",
            "GET /get_reference_files": "List reference files (backward compatibility)",
            "GET /get_predefined_voices": "List predefined voices (backward compatibility)",
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

