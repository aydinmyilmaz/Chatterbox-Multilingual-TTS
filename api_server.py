"""
FastAPI server using the same code as the working Gradio app
This ensures Turkish/Arabic audio quality matches the Gradio app
"""

import os
import io
import random
import numpy as np
import torch
from pathlib import Path
from typing import Optional
import logging

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import soundfile as sf

# Import from Gradio app (same working code)
from src.chatterbox.mtl_tts import ChatterboxMultilingualTTS, SUPPORTED_LANGUAGES

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Device configuration (same as Gradio app)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"🚀 Running on device: {DEVICE}")

# Global model (same as Gradio app)
MODEL = None

# Language config (same as Gradio app)
LANGUAGE_CONFIG = {
    "ar": {
        "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/ar_f/ar_prompts2.flac",
        "text": "في الشهر الماضي، وصلنا إلى معلم جديد بمليارين من المشاهدات على قناتنا على يوتيوب."
    },
    "da": {
        "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/da_m1.flac",
        "text": "Sidste måned nåede vi en ny milepæl med to milliarder visninger på vores YouTube-kanal."
    },
    "de": {
        "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/de_f1.flac",
        "text": "Letzten Monat haben wir einen neuen Meilenstein erreicht: zwei Milliarden Aufrufe auf unserem YouTube-Kanal."
    },
    "el": {
        "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/el_m.flac",
        "text": "Τον περασμένο μήνα, φτάσαμε σε ένα νέο ορόσημο με δύο δισεκατομμύρια προβολές στο κανάλι μας στο YouTube."
    },
    "en": {
        "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/en_f1.flac",
        "text": "Last month, we reached a new milestone with two billion views on our YouTube channel."
    },
    "es": {
        "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/es_f1.flac",
        "text": "El mes pasado alcanzamos un nuevo hito: dos mil millones de visualizaciones en nuestro canal de YouTube."
    },
    "fi": {
        "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/fi_m.flac",
        "text": "Viime kuussa saavutimme uuden virstanpylvään kahden miljardin katselukerran kanssa YouTube-kanavallamme."
    },
    "fr": {
        "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/fr_f1.flac",
        "text": "Le mois dernier, nous avons atteint un nouveau jalon avec deux milliards de vues sur notre chaîne YouTube."
    },
    "he": {
        "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/he_m1.flac",
        "text": "בחודש שעבר הגענו לאבן דרך חדשה עם שני מיליארד צפיות בערוץ היוטיוב שלנו."
    },
    "hi": {
        "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/hi_f1.flac",
        "text": "पिछले महीने हमने एक नया मील का पत्थर छुआ: हमारे YouTube चैनल पर दो अरब व्यूज़।"
    },
    "it": {
        "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/it_m1.flac",
        "text": "Il mese scorso abbiamo raggiunto un nuovo traguardo: due miliardi di visualizzazioni sul nostro canale YouTube."
    },
    "ja": {
        "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/ja/ja_prompts1.flac",
        "text": "先月、私たちのYouTubeチャンネルで二十億回の再生回数という新たなマイルストーンに到達しました。"
    },
    "ko": {
        "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/ko_f.flac",
        "text": "지난달 우리는 유튜브 채널에서 이십억 조회수라는 새로운 이정표에 도달했습니다."
    },
    "ms": {
        "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/ms_f.flac",
        "text": "Bulan lepas, kami mencapai pencapaian baru dengan dua bilion tontonan di saluran YouTube kami."
    },
    "nl": {
        "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/nl_m.flac",
        "text": "Vorige maand bereikten we een nieuwe mijlpaal met twee miljard weergaven op ons YouTube-kanaal."
    },
    "no": {
        "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/no_f1.flac",
        "text": "Forrige måned nådde vi en ny milepæl med to milliarder visninger på YouTube-kanalen vår."
    },
    "pl": {
        "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/pl_m.flac",
        "text": "W zeszłym miesiącu osiągnęliśmy nowy kamień milowy z dwoma miliardami wyświetleń na naszym kanale YouTube."
    },
    "pt": {
        "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/pt_m1.flac",
        "text": "No mês passado, alcançámos um novo marco: dois mil milhões de visualizações no nosso canal do YouTube."
    },
    "ru": {
        "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/ru_m.flac",
        "text": "В прошлом месяце мы достигли нового рубежа: два миллиарда просмотров на нашем YouTube-канале."
    },
    "sv": {
        "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/sv_f.flac",
        "text": "Förra månaden nådde vi en ny milstolpe med två miljarder visningar på vår YouTube-kanal."
    },
    "sw": {
        "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/sw_m.flac",
        "text": "Mwezi uliopita, tulifika hatua mpya ya maoni ya bilioni mbili kweny kituo chetu cha YouTube."
    },
    "tr": {
        "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/tr_m.flac",
        "text": "Geçen ay YouTube kanalımızda iki milyar görüntüleme ile yeni bir dönüm noktasına ulaştık."
    },
    "zh": {
        "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/zh_f2.flac",
        "text": "上个月，我们达到了一个新的里程碑。 我们的YouTube频道观看次数达到了二十亿次，这绝对令人难以置信。"
    },
}

# Helper functions (same as Gradio app)
def get_or_load_model():
    """Loads the ChatterboxMultilingualTTS model if it hasn't been loaded already."""
    global MODEL
    if MODEL is None:
        logger.info("Model not loaded, initializing...")
        try:
            MODEL = ChatterboxMultilingualTTS.from_pretrained(DEVICE)
            if hasattr(MODEL, 'to') and str(MODEL.device) != DEVICE:
                MODEL.to(DEVICE)
            logger.info(f"Model loaded successfully. Internal device: {getattr(MODEL, 'device', 'N/A')}")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    return MODEL

def set_seed(seed: int):
    """Sets the random seed for reproducibility."""
    torch.manual_seed(seed)
    if DEVICE == "cuda":
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)

def default_audio_for_ui(lang: str) -> Optional[str]:
    """Get default audio URL for a language."""
    return LANGUAGE_CONFIG.get(lang, {}).get("audio")

def resolve_audio_prompt(language_id: str, provided_path: Optional[str]) -> Optional[str]:
    """
    Decide which audio prompt to use:
    - If user provided a path, use it (priority).
    - We don't use default audio URLs - only user-uploaded files.
    """
    if provided_path and str(provided_path).strip():
        # Check if it's a local file path
        local_path = Path(provided_path)
        if local_path.exists():
            return str(local_path)
        # Check if it's in reference_audio directory
        ref_path = REFERENCE_AUDIO_DIR / local_path.name
        if ref_path.exists():
            return str(ref_path)
        # Otherwise assume it's a URL or absolute path
        return provided_path
    # Don't use default audio URLs - only user-uploaded files
    return None

# Reference audio storage
REFERENCE_AUDIO_DIR = Path("./reference_audio")
REFERENCE_AUDIO_DIR.mkdir(exist_ok=True)

# FastAPI app
app = FastAPI(title="Chatterbox Multilingual TTS API", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request models
class TTSRequest(BaseModel):
    text: str = Field(..., description="Text to synthesize (max 300 characters)", max_length=300)
    language_id: str = Field(..., description="Language code (e.g., en, tr, ar, fr)")
    audio_prompt_path: Optional[str] = Field(None, description="Path or URL to reference audio file")
    exaggeration: float = Field(0.5, ge=0.25, le=2.0, description="Speech expressiveness (0.25-2.0)")
    temperature: float = Field(0.8, ge=0.05, le=5.0, description="Randomness in generation (0.05-5.0)")
    seed: int = Field(0, description="Random seed (0 for random)")
    cfg_weight: float = Field(0.5, ge=0.0, le=1.0, description="CFG/Pace weight (0.2-1.0)")

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
    model_loaded = MODEL is not None
    return {
        "status": "healthy" if model_loaded else "model_not_loaded",
        "model_loaded": model_loaded,
        "device": DEVICE,
        "cuda_available": torch.cuda.is_available()
    }

# Supported languages
@app.get("/languages")
async def get_languages():
    """Get list of supported languages."""
    return {
        "supported_languages": SUPPORTED_LANGUAGES,
        "count": len(SUPPORTED_LANGUAGES)
    }

# Upload reference audio
@app.post("/upload_reference")
async def upload_reference(files: list[UploadFile] = File(...)):
    """
    Upload reference audio file(s) for voice cloning.
    Files are stored in the reference_audio directory.
    Supported formats: WAV, MP3, FLAC, etc.
    """
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

            # Save file
            with open(destination_path, "wb") as buffer:
                content = await file.read()
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

# Get reference files
@app.get("/reference_files")
async def get_reference_files():
    """Get list of uploaded reference audio files."""
    files = []
    for file_path in REFERENCE_AUDIO_DIR.glob("*"):
        if file_path.is_file() and file_path.suffix.lower() in ['.wav', '.mp3', '.flac', '.ogg', '.m4a']:
            files.append(file_path.name)
    return sorted(files)

# Generate TTS endpoint (using same code as Gradio app)
@app.post("/generate")
async def generate_tts(request: TTSRequest):
    """
    Generate speech audio from text using the same code as the working Gradio app.
    This ensures Turkish/Arabic audio quality matches the Gradio app.
    """
    try:
        # Validate language
        if request.language_id.lower() not in SUPPORTED_LANGUAGES:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported language_id '{request.language_id}'. Supported: {', '.join(SUPPORTED_LANGUAGES.keys())}"
            )

        # Get model (same as Gradio app)
        current_model = get_or_load_model()
        if current_model is None:
            raise HTTPException(status_code=503, detail="TTS model is not loaded")

        # Set seed if provided (same as Gradio app)
        if request.seed != 0:
            set_seed(int(request.seed))

        logger.info(f"Generating audio for text: '{request.text[:50]}...' (lang: {request.language_id})")

        # Handle optional audio prompt (only user-uploaded files, no default URLs)
        chosen_prompt = resolve_audio_prompt(request.language_id, request.audio_prompt_path)

        # Prepare generation kwargs (same as Gradio app)
        generate_kwargs = {
            "exaggeration": request.exaggeration,
            "temperature": request.temperature,
            "cfg_weight": request.cfg_weight,
        }
        if chosen_prompt:
            generate_kwargs["audio_prompt_path"] = chosen_prompt
            logger.info(f"Using audio prompt: {chosen_prompt}")
        else:
            logger.info("No audio prompt provided - model will use its default voice")

        # Generate audio (EXACT same code as Gradio app)
        wav = current_model.generate(
            request.text[:300],  # Truncate text to max chars (same as Gradio)
            language_id=request.language_id,  # Same as Gradio app
            **generate_kwargs
        )

        # Convert to numpy array (same as Gradio app)
        audio_np = wav.squeeze(0).numpy()
        sample_rate = current_model.sr

        logger.info(f"Audio generation complete. Sample rate: {sample_rate}Hz, Length: {len(audio_np)} samples")

        # Create audio buffer
        audio_buffer = io.BytesIO()
        sf.write(audio_buffer, audio_np, sample_rate, format="WAV")
        audio_buffer.seek(0)

        return StreamingResponse(
            audio_buffer,
            media_type="audio/wav",
            headers={
                "Content-Disposition": f'attachment; filename="tts_output.wav"',
                "X-Sample-Rate": str(sample_rate)
            }
        )

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
        "name": "Chatterbox Multilingual TTS API",
        "version": "1.0.0",
        "description": "FastAPI server using the same code as the working Gradio app",
        "endpoints": {
            "POST /generate": "Generate TTS audio",
            "POST /upload_reference": "Upload reference audio file(s)",
            "GET /reference_files": "List uploaded reference audio files",
            "GET /languages": "Get supported languages",
            "GET /health": "Health check"
        },
        "supported_languages_count": len(SUPPORTED_LANGUAGES),
        "note": "Only user-uploaded reference audio files are used. Default audio URLs are not used."
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8004))
    uvicorn.run(app, host="0.0.0.0", port=port)

