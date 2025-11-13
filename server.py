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
from typing import Optional, List, Literal, Dict, Tuple
import logging
import time

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import soundfile as sf

# Optional imports for advanced audio processing
try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False

try:
    import torchaudio
    TORCHAUDIO_AVAILABLE = True
except ImportError:
    TORCHAUDIO_AVAILABLE = False

try:
    from pydub import AudioSegment
    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False

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

# New simplified request model (HF Space style)
class TTSRequest(BaseModel):
    text: str = Field(..., description="Text to synthesize (max 300 characters)", max_length=300)
    language_id: str = Field(..., description="Language code (e.g., 'en', 'tr', 'ar')")
    audio_prompt_path: Optional[str] = Field(None, description="Path to reference audio file (optional)")
    exaggeration: float = Field(0.5, ge=0.25, le=2.0, description="Speech expressiveness (0.25-2.0)")
    temperature: float = Field(0.8, ge=0.05, le=5.0, description="Randomness in generation (0.05-5.0)")
    seed: int = Field(0, description="Random seed (0 for random)")
    cfg_weight: float = Field(0.5, ge=0.2, le=1.0, description="CFG/Pace weight (0.2-1.0)")

# Legacy request model (Chatterbox-TTS-Server compatibility)
class CustomTTSRequest(BaseModel):
    """Request model for the custom /tts endpoint (backward compatibility)."""
    text: str = Field(..., min_length=1, description="Text to be synthesized.")
    voice_mode: Literal["predefined", "clone"] = Field("predefined", description="Voice mode")
    predefined_voice_id: Optional[str] = Field(None, description="Filename of predefined voice")
    reference_audio_filename: Optional[str] = Field(None, description="Filename of reference audio for cloning")
    output_format: Optional[Literal["wav", "opus", "mp3"]] = Field("wav", description="Audio output format")
    split_text: Optional[bool] = Field(True, description="Whether to split long text into chunks")
    chunk_size: Optional[int] = Field(120, ge=50, le=500, description="Target chunk size")
    temperature: Optional[float] = Field(None, description="Temperature override")
    exaggeration: Optional[float] = Field(None, description="Exaggeration override")
    cfg_weight: Optional[float] = Field(None, description="CFG weight override")
    seed: Optional[int] = Field(None, description="Seed override")
    speed_factor: Optional[float] = Field(None, description="Speed factor override")
    language: Optional[str] = Field(None, description="Language code override")

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

def chunk_text_by_sentences(full_text: str, chunk_size: int) -> List[str]:
    """
    Chunks text into manageable pieces for TTS processing, respecting sentence boundaries.
    Simplified version matching old server behavior.
    """
    if not full_text or full_text.isspace():
        return []
    if chunk_size <= 0:
        chunk_size = float("inf")

    # Simple sentence splitting by common punctuation
    sentences = []
    current_sentence = ""
    for char in full_text:
        current_sentence += char
        if char in '.!?':
            sentences.append(current_sentence.strip())
            current_sentence = ""
    if current_sentence.strip():
        sentences.append(current_sentence.strip())

    # If no punctuation found, treat entire text as one sentence
    if not sentences:
        sentences = [full_text]

    # Group sentences into chunks
    text_chunks: List[str] = []
    current_chunk_sentences: List[str] = []
    current_chunk_length = 0

    for sentence in sentences:
        sentence_len = len(sentence)
        if not current_chunk_sentences:
            current_chunk_sentences.append(sentence)
            current_chunk_length = sentence_len
        elif current_chunk_length + 1 + sentence_len <= chunk_size:
            current_chunk_sentences.append(sentence)
            current_chunk_length += 1 + sentence_len
        else:
            if current_chunk_sentences:
                text_chunks.append(" ".join(current_chunk_sentences))
            current_chunk_sentences = [sentence]
            current_chunk_length = sentence_len

        # If single sentence exceeds chunk_size, it forms its own chunk
        if current_chunk_length > chunk_size and len(current_chunk_sentences) == 1:
            text_chunks.append(" ".join(current_chunk_sentences))
            current_chunk_sentences = []
            current_chunk_length = 0

    if current_chunk_sentences:
        text_chunks.append(" ".join(current_chunk_sentences))

    text_chunks = [chunk for chunk in text_chunks if chunk.strip()]

    if not text_chunks and full_text.strip():
        logger.warning("Text chunking resulted in zero chunks. Returning full text as one chunk.")
        return [full_text.strip()]

    logger.info(f"Text chunking complete. Generated {len(text_chunks)} chunk(s).")
    return text_chunks

def apply_speed_factor(audio_tensor: torch.Tensor, sample_rate: int, speed_factor: float) -> Tuple[torch.Tensor, int]:
    """
    Applies a speed factor to an audio tensor.
    Uses librosa.effects.time_stretch if available for pitch preservation.
    """
    if speed_factor == 1.0:
        return audio_tensor, sample_rate
    if speed_factor <= 0:
        logger.warning(f"Invalid speed_factor {speed_factor}. Must be positive. Returning original audio.")
        return audio_tensor, sample_rate

    audio_tensor_cpu = audio_tensor.cpu()
    # Ensure tensor is 1D mono
    if audio_tensor_cpu.ndim == 2:
        if audio_tensor_cpu.shape[0] == 1:
            audio_tensor_cpu = audio_tensor_cpu.squeeze(0)
        elif audio_tensor_cpu.shape[1] == 1:
            audio_tensor_cpu = audio_tensor_cpu.squeeze(1)
        else:
            logger.warning(f"Multi-channel audio received. Using first channel only.")
            audio_tensor_cpu = audio_tensor_cpu[0, :]

    if audio_tensor_cpu.ndim != 1:
        logger.error(f"apply_speed_factor: audio_tensor_cpu is not 1D. Returning original audio.")
        return audio_tensor, sample_rate

    if LIBROSA_AVAILABLE:
        try:
            audio_np = audio_tensor_cpu.numpy()
            stretched_audio_np = librosa.effects.time_stretch(y=audio_np, rate=speed_factor)
            speed_adjusted_tensor = torch.from_numpy(stretched_audio_np)
            logger.info(f"Applied speed factor {speed_factor} using librosa.")
            return speed_adjusted_tensor, sample_rate
        except Exception as e:
            logger.error(f"Failed to apply speed factor using librosa: {e}. Returning original audio.")
            return audio_tensor, sample_rate
    else:
        logger.warning(f"Librosa not available for speed adjustment. Returning original audio.")
        return audio_tensor, sample_rate

def encode_audio(
    audio_array: np.ndarray,
    sample_rate: int,
    output_format: str = "wav",
    target_sample_rate: Optional[int] = None,
) -> Optional[bytes]:
    """
    Encodes a NumPy audio array into the specified format (WAV, Opus, or MP3).
    """
    if audio_array is None or audio_array.size == 0:
        logger.warning("encode_audio received empty or None audio array.")
        return None

    # Ensure audio is float32
    if audio_array.dtype != np.float32:
        if np.issubdtype(audio_array.dtype, np.integer):
            max_val = np.iinfo(audio_array.dtype).max
            audio_array = audio_array.astype(np.float32) / max_val
        else:
            audio_array = audio_array.astype(np.float32)

    # Ensure audio is mono
    if audio_array.ndim == 2 and audio_array.shape[1] == 1:
        audio_array = audio_array.squeeze(axis=1)
    elif audio_array.ndim > 1:
        logger.warning(f"Multi-channel audio provided. Using only the first channel.")
        audio_array = audio_array[:, 0]

    # Resample if target_sample_rate is provided
    if target_sample_rate is not None and target_sample_rate != sample_rate and LIBROSA_AVAILABLE:
        try:
            logger.info(f"Resampling audio from {sample_rate}Hz to {target_sample_rate}Hz.")
            audio_array = librosa.resample(y=audio_array, orig_sr=sample_rate, target_sr=target_sample_rate)
            sample_rate = target_sample_rate
        except Exception as e:
            logger.error(f"Error resampling audio: {e}. Proceeding with original sample rate.")

    output_buffer = io.BytesIO()

    try:
        if output_format == "opus":
            OPUS_SUPPORTED_RATES = {8000, 12000, 16000, 24000, 48000}
            TARGET_OPUS_RATE = 48000

            rate_to_write = sample_rate
            if rate_to_write not in OPUS_SUPPORTED_RATES:
                if LIBROSA_AVAILABLE:
                    logger.warning(f"Resampling to {TARGET_OPUS_RATE}Hz for Opus encoding.")
                    audio_array = librosa.resample(y=audio_array, orig_sr=rate_to_write, target_sr=TARGET_OPUS_RATE)
                    rate_to_write = TARGET_OPUS_RATE
            sf.write(output_buffer, audio_array, rate_to_write, format="ogg", subtype="opus")

        elif output_format == "wav":
            audio_clipped = np.clip(audio_array, -1.0, 1.0)
            audio_int16 = (audio_clipped * 32767).astype(np.int16)
            sf.write(output_buffer, audio_int16, sample_rate, format="wav", subtype="pcm_16")

        elif output_format == "mp3":
            if not PYDUB_AVAILABLE:
                logger.error("pydub not available for MP3 encoding. Falling back to WAV.")
                audio_clipped = np.clip(audio_array, -1.0, 1.0)
                audio_int16 = (audio_clipped * 32767).astype(np.int16)
                sf.write(output_buffer, audio_int16, sample_rate, format="wav", subtype="pcm_16")
            else:
                audio_clipped = np.clip(audio_array, -1.0, 1.0)
                audio_int16 = (audio_clipped * 32767).astype(np.int16)
                audio_segment = AudioSegment(
                    audio_int16.tobytes(),
                    frame_rate=sample_rate,
                    sample_width=2,
                    channels=1,
                )
                audio_segment.export(output_buffer, format="mp3")

        else:
            logger.error(f"Unsupported output format: {output_format}")
            return None

        encoded_bytes = output_buffer.getvalue()
        logger.info(f"Encoded {len(encoded_bytes)} bytes to '{output_format}' at {sample_rate}Hz.")
        return encoded_bytes

    except Exception as e:
        logger.error(f"Error encoding audio to '{output_format}': {e}", exc_info=True)
        return None

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

@app.get("/get_reference_files", response_model=List[str], tags=["UI Helpers"])
async def get_reference_files_api():
    """
    Returns a list of valid reference audio filenames (.wav, .mp3).
    Backward compatibility endpoint for Chatterbox-TTS-Server.
    """
    logger.debug("Request for /get_reference_files.")
    try:
        files = []
        for file_path in REFERENCE_AUDIO_DIR.glob("*"):
            if file_path.is_file() and file_path.suffix.lower() in ['.wav', '.mp3']:
                files.append(file_path.name)
        return sorted(files)
    except Exception as e:
        logger.error(f"Error getting reference files for API: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail="Failed to retrieve reference audio files."
        )

@app.get("/get_predefined_voices", response_model=List[Dict[str, str]], tags=["UI Helpers"])
async def get_predefined_voices_api():
    """
    Returns a list of predefined voices with display names and filenames.
    Backward compatibility endpoint for Chatterbox-TTS-Server.
    """
    logger.debug("Request for /get_predefined_voices.")
    try:
        voices = []
        for file_path in REFERENCE_AUDIO_DIR.glob("*"):
            if file_path.is_file() and file_path.suffix.lower() in ['.wav', '.mp3']:
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

@app.post("/upload_predefined_voice", tags=["File Management"])
async def upload_predefined_voice_endpoint(files: List[UploadFile] = File(...)):
    """
    Upload predefined voice files (backward compatibility with Chatterbox-TTS-Server).
    Files are stored in the reference_audio directory (simplified structure).
    """
    uploaded_filenames_successfully: List[str] = []
    upload_errors: List[Dict[str, str]] = []

    for file in files:
        if not file.filename:
            upload_errors.append(
                {"filename": "Unknown", "error": "File received with no filename."}
            )
            logger.warning("Upload attempt for predefined voice with no filename.")
            continue

        # Sanitize filename
        safe_filename = file.filename.replace(" ", "_").replace("/", "_").replace("\\", "_")
        destination_path = REFERENCE_AUDIO_DIR / safe_filename

        try:
            # Validate file type
            if not (
                safe_filename.lower().endswith(".wav")
                or safe_filename.lower().endswith(".mp3")
            ):
                raise ValueError(
                    "Invalid file type. Only .wav and .mp3 are allowed for predefined voices."
                )

            if destination_path.exists():
                logger.info(
                    f"Predefined voice file '{safe_filename}' already exists. Skipping duplicate upload."
                )
                if safe_filename not in uploaded_filenames_successfully:
                    uploaded_filenames_successfully.append(safe_filename)
                continue

            # Save file
            with open(destination_path, "wb") as buffer:
                content = await file.read()
                buffer.write(content)

            logger.info(
                f"Successfully saved uploaded predefined voice file to: {destination_path}"
            )
            uploaded_filenames_successfully.append(safe_filename)

        except Exception as e_upload:
            error_msg = f"Error processing predefined voice file '{file.filename}': {str(e_upload)}"
            logger.error(error_msg, exc_info=True)
            upload_errors.append({"filename": file.filename, "error": str(e_upload)})
        finally:
            await file.close()

    # Get all current reference audio files (as predefined voices)
    all_files = []
    for file_path in REFERENCE_AUDIO_DIR.glob("*"):
        if file_path.is_file():
            all_files.append({
                "id": file_path.name,
                "filename": file_path.name,
                "path": str(file_path.relative_to(Path(".")))
            })

    response_data = {
        "message": f"Processed {len(files)} predefined voice file(s).",
        "uploaded_files": uploaded_filenames_successfully,
        "all_predefined_voices": all_files,
        "errors": upload_errors,
    }
    status_code = (
        200 if not upload_errors or len(uploaded_filenames_successfully) > 0 else 400
    )
    if upload_errors:
        logger.warning(
            f"Upload to /upload_predefined_voice completed with {len(upload_errors)} error(s)."
        )
    return JSONResponse(content=response_data, status_code=status_code)

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

@app.post("/tts", tags=["TTS"], summary="Legacy /tts endpoint (backward compatibility)")
async def legacy_tts_endpoint(request: CustomTTSRequest):
    """
    Legacy /tts endpoint for backward compatibility with Chatterbox-TTS-Server.
    Fully compatible with old server: handles chunking, speed_factor, and output_format.
    """
    try:
        # Check if model is loaded
        model = get_or_load_model()
        if model is None:
            raise HTTPException(status_code=503, detail="TTS model is not loaded")

        logger.info(f"Received /tts request: mode='{request.voice_mode}', format='{request.output_format}'")
        logger.debug(f"TTS params: seed={request.seed}, split={request.split_text}, chunk_size={request.chunk_size}")

        # Resolve audio prompt path based on voice_mode (same as old server)
        audio_prompt_path_for_engine: Optional[str] = None
        if request.voice_mode == "predefined":
            if not request.predefined_voice_id:
                raise HTTPException(
                    status_code=400,
                    detail="Missing 'predefined_voice_id' for 'predefined' voice mode."
                )
            predefined_path = REFERENCE_AUDIO_DIR / request.predefined_voice_id
            if not predefined_path.is_file():
                raise HTTPException(
                    status_code=404,
                    detail=f"Predefined voice file '{request.predefined_voice_id}' not found."
                )
            audio_prompt_path_for_engine = str(predefined_path)
            logger.info(f"Using predefined voice: {request.predefined_voice_id}")

        elif request.voice_mode == "clone":
            if not request.reference_audio_filename:
                raise HTTPException(
                    status_code=400,
                    detail="Missing 'reference_audio_filename' for 'clone' voice mode."
                )
            clone_path = REFERENCE_AUDIO_DIR / request.reference_audio_filename
            if not clone_path.is_file():
                raise HTTPException(
                    status_code=404,
                    detail=f"Reference audio file '{request.reference_audio_filename}' not found."
                )
            audio_prompt_path_for_engine = str(clone_path)
            logger.info(f"Using reference audio for cloning: {request.reference_audio_filename}")

        # Handle text chunking (same logic as old server)
        all_audio_segments_np: List[np.ndarray] = []
        engine_output_sample_rate: Optional[int] = None

        if request.split_text and len(request.text) > (request.chunk_size * 1.5 if request.chunk_size else 120 * 1.5):
            chunk_size_to_use = request.chunk_size if request.chunk_size is not None else 120
            logger.info(f"Splitting text into chunks of size ~{chunk_size_to_use}.")
            text_chunks = chunk_text_by_sentences(request.text, chunk_size_to_use)
        else:
            text_chunks = [request.text]
            logger.info("Processing text as a single chunk (splitting not enabled or text too short).")

        if not text_chunks:
            raise HTTPException(status_code=400, detail="Text processing resulted in no usable chunks.")

        # Process each chunk (same as old server)
        for i, chunk in enumerate(text_chunks):
            logger.info(f"Synthesizing chunk {i+1}/{len(text_chunks)}...")
            try:
                # Set seed if provided
                seed_to_use = request.seed if request.seed is not None else 0
                if seed_to_use != 0:
                    set_seed(seed_to_use)

                # Get language
                language_to_use = request.language or "en"

                # Prepare generation kwargs
                generate_kwargs = {
                    "language_id": language_to_use.lower(),
                    "exaggeration": request.exaggeration if request.exaggeration is not None else 0.5,
                    "temperature": request.temperature if request.temperature is not None else 0.8,
                    "cfg_weight": request.cfg_weight if request.cfg_weight is not None else 0.5,
                }
                if audio_prompt_path_for_engine:
                    generate_kwargs["audio_prompt_path"] = audio_prompt_path_for_engine

                # Generate audio for this chunk
                wav_tensor = model.generate(
                    text=chunk[:300],  # Max 300 chars per chunk
                    **generate_kwargs
                )

                if wav_tensor is None:
                    error_detail = f"TTS engine failed to synthesize audio for chunk {i+1}."
                    logger.error(error_detail)
                    raise HTTPException(status_code=500, detail=error_detail)

                chunk_sr_from_engine = model.sr
                if engine_output_sample_rate is None:
                    engine_output_sample_rate = chunk_sr_from_engine
                elif engine_output_sample_rate != chunk_sr_from_engine:
                    logger.warning(
                        f"Inconsistent sample rate from engine: chunk {i+1} ({chunk_sr_from_engine}Hz) "
                        f"differs from previous ({engine_output_sample_rate}Hz). Using first chunk's SR."
                    )

                current_processed_audio_tensor = wav_tensor

                # Apply speed_factor if provided (same as old server)
                speed_factor_to_use = request.speed_factor if request.speed_factor is not None else 1.0
                if speed_factor_to_use != 1.0:
                    current_processed_audio_tensor, _ = apply_speed_factor(
                        current_processed_audio_tensor,
                        chunk_sr_from_engine,
                        speed_factor_to_use,
                    )

                # Convert to numpy and add to segments
                processed_audio_np = current_processed_audio_tensor.cpu().numpy().squeeze()
                all_audio_segments_np.append(processed_audio_np)

            except HTTPException:
                raise
            except Exception as e_chunk:
                error_detail = f"Error processing audio chunk {i+1}: {str(e_chunk)}"
                logger.error(error_detail, exc_info=True)
                raise HTTPException(status_code=500, detail=error_detail)

        if not all_audio_segments_np:
            logger.error("No audio segments were successfully generated.")
            raise HTTPException(status_code=500, detail="Audio generation resulted in no output.")

        if engine_output_sample_rate is None:
            logger.error("Engine output sample rate could not be determined.")
            raise HTTPException(status_code=500, detail="Failed to determine engine sample rate.")

        # Concatenate all chunks (same as old server)
        try:
            final_audio_np = (
                np.concatenate(all_audio_segments_np)
                if len(all_audio_segments_np) > 1
                else all_audio_segments_np[0]
            )
        except ValueError as e_concat:
            logger.error(f"Audio concatenation failed: {e_concat}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Audio concatenation error: {e_concat}")

        # Encode audio to requested format (same as old server)
        output_format_str = request.output_format if request.output_format else "wav"
        encoded_audio_bytes = encode_audio(
            audio_array=final_audio_np,
            sample_rate=engine_output_sample_rate,
            output_format=output_format_str,
            target_sample_rate=None,  # Keep original sample rate
        )

        if encoded_audio_bytes is None or len(encoded_audio_bytes) < 100:
            logger.error(f"Failed to encode final audio to format: {output_format_str}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to encode audio to {output_format_str} or generated invalid audio."
            )

        media_type = f"audio/{output_format_str}"
        timestamp_str = time.strftime("%Y%m%d_%H%M%S")
        suggested_filename_base = f"tts_output_{timestamp_str}"
        download_filename = f"{suggested_filename_base}.{output_format_str}"
        headers = {"Content-Disposition": f'attachment; filename="{download_filename}"'}

        logger.info(f"Successfully generated audio: {download_filename}, {len(encoded_audio_bytes)} bytes, type {media_type}.")

        return StreamingResponse(
            io.BytesIO(encoded_audio_bytes), media_type=media_type, headers=headers
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in legacy /tts endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to generate audio: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8004))  # Default port 8004 (same as Chatterbox-TTS-Server)
    uvicorn.run(app, host="0.0.0.0", port=port)

