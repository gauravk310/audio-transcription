import asyncio
from datetime import datetime, timezone
import logging
import os
from pathlib import Path
import shutil
import tempfile
import time
from typing import List, Literal, Optional

from fastapi import FastAPI, File, Form, HTTPException, Query, Request, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from transcribe import DEFAULT_MODEL_SIZE, download_audio_from_url, transcribe_audio

logger = logging.getLogger(__name__)

# Application metadata
APP_NAME = "Audio Transcription Service"
APP_VERSION = "0.3.0"
APP_DESCRIPTION = "FastAPI service for audio file transcription to text using Whisper. Supports direct file uploads and audio URLs."

# Record server start time for uptime tracking
START_TIME = time.time()

# Allowed audio extensions
ALLOWED_EXTENSIONS = {
    ".mp3", ".wav", ".m4a", ".ogg", ".flac", ".aac",
    ".wma", ".webm", ".opus", ".mp4", ".mkv",
}

app = FastAPI(
    title=APP_NAME,
    version=APP_VERSION,
    description=APP_DESCRIPTION,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS Middleware setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class HealthResponse(BaseModel):
    """Schema for the health check response."""
    status: Literal["ok", "degraded", "error"] = Field(
        default="ok",
        description="Current health status of the service",
        examples=["ok"],
    )
    service: str = Field(
        default=APP_NAME,
        description="Name of the service",
        examples=[APP_NAME],
    )
    version: str = Field(
        default=APP_VERSION,
        description="Current version of the application",
        examples=[APP_VERSION],
    )
    uptime_seconds: float = Field(
        description="Service uptime in seconds",
        examples=[124.52],
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC timestamp of the health check",
    )
    environment: str = Field(
        default_factory=lambda: os.getenv("ENVIRONMENT", "development"),
        description="Deployment environment",
        examples=["development", "production"],
    )


class SegmentResponse(BaseModel):
    """Timestamped segment schema."""
    id: int = Field(description="Segment sequence index")
    start: float = Field(description="Segment start time in seconds")
    end: float = Field(description="Segment end time in seconds")
    text: str = Field(description="Transcribed segment text")


class TranscriptionResponse(BaseModel):
    """Schema for audio transcription response."""
    success: bool = Field(default=True, description="Whether the transcription succeeded")
    filename: str = Field(description="Original uploaded filename or URL audio name")
    source: Literal["upload", "url"] = Field(default="upload", description="Source of the audio: 'upload' or 'url'")
    text: str = Field(description="Full transcribed text")
    language: Optional[str] = Field(default=None, description="Detected or specified language code")
    language_probability: Optional[float] = Field(default=None, description="Language detection confidence (0.0 to 1.0)")
    duration: Optional[float] = Field(default=None, description="Audio duration in seconds")
    processing_time_seconds: float = Field(description="Time taken to process and transcribe the audio")
    model: str = Field(description="Whisper model size used")
    engine: str = Field(description="Transcription engine used")
    segments: Optional[List[SegmentResponse]] = Field(default=None, description="Optional timestamped segments")


@app.get("/", tags=["Root"])
def read_root():
    """Root endpoint providing service information and links."""
    return {
        "message": f"Welcome to {APP_NAME}!",
        "version": APP_VERSION,
        "docs": "/docs",
        "health": "/health",
        "endpoints": {
            "transcribe": "POST /transcribe (supports file upload and fileURL parameter)",
        },
    }


@app.get(
    "/health",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    tags=["Health"],
    summary="Health Check",
    description="Returns service health status, uptime, and metadata.",
)
def get_health() -> HealthResponse:
    """Check the health status of the service."""
    uptime = round(time.time() - START_TIME, 2)
    return HealthResponse(
        status="ok",
        service=APP_NAME,
        version=APP_VERSION,
        uptime_seconds=uptime,
        environment=os.getenv("ENVIRONMENT", "development"),
    )


@app.get(
    "/healthz",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    tags=["Health"],
    include_in_schema=False,
)
def get_healthz() -> HealthResponse:
    """Kubernetes / Container liveness probe alias."""
    return get_health()


@app.post(
    "/transcribe",
    response_model=TranscriptionResponse,
    status_code=status.HTTP_200_OK,
    tags=["Transcription"],
    summary="Transcribe Audio File or URL",
    description=(
        "Converts an audio file to text using Whisper. "
        "Provide either a file upload via `file` OR a public audio URL via `fileURL`."
    ),
)
async def transcribe_endpoint(
    request: Request,
    file: Optional[UploadFile] = File(None, description="Audio file to upload directly (e.g. mp3, wav, m4a, flac)"),
    fileURL: Optional[str] = Form(None, description="Public URL of the audio file to download and transcribe"),
    file_url: Optional[str] = Form(None, description="Alias for fileURL"),
    model: Optional[str] = Form(None, description="Model size: 'tiny', 'base', 'small', 'medium', 'large-v3' (default: medium)"),
    language: Optional[str] = Form(None, description="Audio language code (e.g. 'en', 'es'). Auto-detected if omitted."),
    include_segments: bool = Form(False, description="Whether to include timestamped segments in the response."),
    fileURL_query: Optional[str] = Query(None, alias="fileURL", include_in_schema=False),
    file_url_query: Optional[str] = Query(None, alias="file_url", include_in_schema=False),
) -> TranscriptionResponse:
    """Endpoint that receives either an uploaded audio file or audio URL, transcribes it, and returns the text."""
    target_url = fileURL or file_url or fileURL_query or file_url_query
    has_file = file is not None and bool(file.filename)

    # If request is sent as application/json instead of form-data
    content_type = request.headers.get("content-type", "")
    if "application/json" in content_type and not target_url and not has_file:
        try:
            body = await request.json()
            target_url = body.get("fileURL") or body.get("file_url")
            model = body.get("model") or model
            language = body.get("language") or language
            include_segments = body.get("include_segments", include_segments)
        except Exception:
            pass

    # Ensure at least one audio input source is provided
    if not has_file and not target_url:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Either an audio 'file' upload or a 'fileURL' parameter must be provided.",
        )

    temp_file = None
    source_type: Literal["upload", "url"] = "upload"
    resolved_filename = ""

    try:
        if has_file:
            # Handle direct file upload
            resolved_filename = file.filename  # type: ignore[union-attr]
            file_ext = Path(resolved_filename).suffix.lower()

            if file_ext and file_ext not in ALLOWED_EXTENSIONS:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=(
                        f"Unsupported file format '{file_ext}'. "
                        f"Allowed formats: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
                    ),
                )

            with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext or ".mp3") as tmp:
                temp_file = tmp.name
                shutil.copyfileobj(file.file, tmp)  # type: ignore[union-attr]
            source_type = "upload"

        else:
            # Handle audio URL download
            source_type = "url"
            logger.info("Fetching audio from URL: %s", target_url)
            try:
                temp_file, resolved_filename = await asyncio.to_thread(
                    download_audio_from_url,
                    target_url.strip(),  # type: ignore[union-attr]
                )
            except ValueError as ve:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=str(ve),
                ) from ve
            except Exception as dl_err:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Failed to download audio from fileURL: {str(dl_err)}",
                ) from dl_err

            file_ext = Path(resolved_filename).suffix.lower()
            if file_ext and file_ext not in ALLOWED_EXTENSIONS:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=(
                        f"Unsupported audio format '{file_ext}' downloaded from URL. "
                        f"Allowed formats: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
                    ),
                )

        # Transcribe audio file
        chosen_model = model.strip() if model else DEFAULT_MODEL_SIZE
        start_time = time.time()
        result = await asyncio.to_thread(
            transcribe_audio,
            file_path=temp_file,
            model_size=chosen_model,
            language=language.strip() if language else None,
        )
        processing_time = round(time.time() - start_time, 2)

        segments = None
        if include_segments and result.get("segments"):
            segments = [
                SegmentResponse(
                    id=seg["id"],
                    start=seg["start"],
                    end=seg["end"],
                    text=seg["text"],
                )
                for seg in result["segments"]
            ]

        return TranscriptionResponse(
            success=True,
            filename=resolved_filename,
            source=source_type,
            text=result.get("text", ""),
            language=result.get("language"),
            language_probability=result.get("language_probability"),
            duration=result.get("duration"),
            processing_time_seconds=processing_time,
            model=result.get("model", chosen_model),
            engine=result.get("engine", "unknown"),
            segments=segments,
        )

    except HTTPException:
        raise
    except RuntimeError as re:
        logger.error("Transcription dependency error: %s", re)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(re),
        ) from re
    except Exception as exc:
        logger.exception("Transcription processing failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Transcription error: {str(exc)}",
        ) from exc
    finally:
        # Clean up temporary file
        if temp_file and os.path.exists(temp_file):
            try:
                os.remove(temp_file)
            except OSError as err:
                logger.warning("Failed to clean up temporary file '%s': %s", temp_file, err)


if __name__ == "__main__":
    import uvicorn

    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    reload = os.getenv("ENVIRONMENT", "development") == "development"

    print(f"Starting {APP_NAME} on http://{host}:{port} (reload={reload})...")
    uvicorn.run("main:app", host=host, port=port, reload=reload)
