"""
FastAPI server for Chatterbox Multilingual TTS
Converted from Gradio app for RunPod deployment
"""

import os
import io
import random
import numpy as np
import torch
from pathlib import Path
from typing import Optional, List
import logging

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import soundfile as sf

from src.chatterbox.mtl_tts import ChatterboxMultilingualTTS, SUPPORTED_LANGUAGES

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Device configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"🚀 Running on device: {DEVICE}")

# Global model
MODEL = None

# Reference audio storage
REFERENCE_AUDIO_DIR = Path("./reference_audio")
REFERENCE_AUDIO_DIR.mkdir(exist_ok=True)

# Language config with default audio URLs
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

# Initialize FastAPI app
app = FastAPI(
    title="Chatterbox Multilingual TTS API",
    description="Generate high-quality multilingual speech from text with reference audio styling",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response models
class TTSRequest(BaseModel):
    text: str = Field(..., description="Text to synthesize (max 300 characters)", max_length=300)
    language_id: str = Field(..., description="Language code (e.g., 'en', 'tr', 'ar')")
    audio_prompt_path: Optional[str] = Field(None, description="Path to reference audio file (optional)")
    exaggeration: float = Field(0.5, ge=0.25, le=2.0, description="Speech expressiveness (0.25-2.0)")
    temperature: float = Field(0.8, ge=0.05, le=5.0, description="Randomness in generation (0.05-5.0)")
    seed: int = Field(0, description="Random seed (0 for random)")
    cfg_weight: float = Field(0.5, ge=0.2, le=1.0, description="CFG/Pace weight (0.2-1.0)")

class LanguageResponse(BaseModel):
    languages: dict
    count: int

class HealthResponse(BaseModel):
    status: str
    device: str
    model_loaded: bool

# Helper functions
def get_or_load_model():
    """Loads the ChatterboxMultilingualTTS model if it hasn't been loaded already,
    and ensures it's on the correct device."""
    global MODEL
    if MODEL is None:
        logger.info("Model not loaded, initializing...")
        try:
            # HF Space uses DEVICE as string, but from_pretrained expects torch.device
            # However, it can accept string too, so we'll use string like Gradio
            MODEL = ChatterboxMultilingualTTS.from_pretrained(DEVICE)
            # Ensure model is on correct device (same as Gradio)
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

def resolve_audio_prompt(language_id: str, provided_path: Optional[str]) -> Optional[str]:
    """Decide which audio prompt to use."""
    if provided_path and str(provided_path).strip():
        # Check if it's a local file path
        local_path = REFERENCE_AUDIO_DIR / provided_path
        if local_path.exists():
            return str(local_path)
        # Otherwise assume it's a URL or absolute path
        return provided_path
    # Fall back to language-specific default
    return LANGUAGE_CONFIG.get(language_id, {}).get("audio")

# Load model at startup
@app.on_event("startup")
async def startup_event():
    try:
        get_or_load_model()
        logger.info("✅ Model loaded successfully at startup")
    except Exception as e:
        logger.error(f"❌ Failed to load model on startup: {e}")

# API Endpoints
@app.get("/", tags=["Info"])
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Chatterbox Multilingual TTS API",
        "version": "1.0.0",
        "description": "Generate high-quality multilingual speech from text",
        "supported_languages": len(SUPPORTED_LANGUAGES),
        "endpoints": {
            "POST /generate": "Generate speech from text",
            "POST /upload_reference": "Upload reference audio file",
            "GET /languages": "Get supported languages",
            "GET /health": "Health check",
            "GET /references": "List uploaded reference audio files"
        }
    }

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy" if MODEL is not None else "model_not_loaded",
        device=DEVICE,
        model_loaded=MODEL is not None
    )

@app.get("/languages", response_model=LanguageResponse, tags=["Languages"])
async def get_languages():
    """Get list of supported languages."""
    return LanguageResponse(
        languages=SUPPORTED_LANGUAGES,
        count=len(SUPPORTED_LANGUAGES)
    )

@app.get("/references", tags=["References"])
async def list_references():
    """List all uploaded reference audio files."""
    files = []
    for file_path in REFERENCE_AUDIO_DIR.glob("*"):
        if file_path.is_file():
            files.append({
                "filename": file_path.name,
                "size": file_path.stat().st_size,
                "path": str(file_path.relative_to(Path(".")))
            })
    return {"files": files, "count": len(files)}

@app.post("/upload_reference", tags=["References"])
async def upload_reference(
    file: UploadFile = File(..., description="Audio file to upload"),
    name: Optional[str] = Form(None, description="Custom filename (optional)")
):
    """
    Upload a reference audio file for voice cloning.
    Supported formats: WAV, MP3, FLAC, etc.
    """
    try:
        # Validate file type
        if not file.content_type or not file.content_type.startswith("audio/"):
            raise HTTPException(status_code=400, detail="File must be an audio file")

        # Determine filename
        filename = name or file.filename
        if not filename:
            raise HTTPException(status_code=400, detail="Filename is required")

        # Save file
        file_path = REFERENCE_AUDIO_DIR / filename
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)

        logger.info(f"Uploaded reference audio: {filename} ({len(content)} bytes)")

        return {
            "status": "success",
            "filename": filename,
            "path": str(file_path.relative_to(Path("."))),
            "size": len(content)
        }
    except Exception as e:
        logger.error(f"Error uploading reference audio: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to upload file: {str(e)}")

@app.post("/generate", tags=["TTS"])
async def generate_tts(request: TTSRequest):
    """
    Generate speech audio from text.

    This endpoint synthesizes natural-sounding speech from input text.
    When a reference audio file is provided, it captures the speaker's voice characteristics.
    """
    try:
        # Validate language
        if request.language_id.lower() not in SUPPORTED_LANGUAGES:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported language_id '{request.language_id}'. Supported: {', '.join(SUPPORTED_LANGUAGES.keys())}"
            )

        # Get model
        model = get_or_load_model()
        if model is None:
            raise HTTPException(status_code=503, detail="TTS model is not loaded")

        # Set seed if provided
        if request.seed != 0:
            set_seed(request.seed)

        logger.info(f"Generating audio for text: '{request.text[:50]}...' (lang: {request.language_id})")

        # Resolve audio prompt
        audio_prompt = resolve_audio_prompt(request.language_id, request.audio_prompt_path)

        # Prepare generation kwargs
        generate_kwargs = {
            "exaggeration": request.exaggeration,
            "temperature": request.temperature,
            "cfg_weight": request.cfg_weight,
        }
        if audio_prompt:
            generate_kwargs["audio_prompt_path"] = audio_prompt
            logger.info(f"Using audio prompt: {audio_prompt}")
        else:
            logger.info("No audio prompt provided; using default voice")

        # Generate audio (same as Gradio - language_id passed directly, model handles lowercase internally)
        wav = model.generate(
            request.text[:300],  # Truncate text to max chars
            language_id=request.language_id,  # Model handles lowercase internally
            **generate_kwargs
        )

        # Convert to numpy array
        audio_np = wav.squeeze(0).numpy()
        sample_rate = model.sr

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

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8004))  # Default port 8004 (same as Chatterbox-TTS-Server)
    uvicorn.run(app, host="0.0.0.0", port=port)

