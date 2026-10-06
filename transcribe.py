"""Audio transcription module and CLI script.

Supports fast, local transcription using `faster-whisper` (recommended)
with fallback to standard `openai-whisper`. Supports both local audio files
and direct audio URLs.
"""

import argparse
import logging
import os
import re
import sys
import tempfile
import time
from typing import Any, Dict, List, Optional, Tuple
import urllib.parse
import urllib.request

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Global model cache to avoid reloading models on every request
_LOADED_MODELS: Dict[str, Any] = {}

DEFAULT_MODEL_SIZE = os.getenv("WHISPER_MODEL", "medium")
DEFAULT_DEVICE = os.getenv("WHISPER_DEVICE", "cpu")
DEFAULT_COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "int8")

MIME_TO_EXTENSION = {
    "audio/mpeg": ".mp3",
    "audio/mp3": ".mp3",
    "audio/wav": ".wav",
    "audio/x-wav": ".wav",
    "audio/wave": ".wav",
    "audio/mp4": ".m4a",
    "audio/x-m4a": ".m4a",
    "audio/m4a": ".m4a",
    "audio/aac": ".aac",
    "audio/ogg": ".ogg",
    "application/ogg": ".ogg",
    "audio/flac": ".flac",
    "audio/x-flac": ".flac",
    "audio/webm": ".webm",
    "video/webm": ".webm",
    "audio/opus": ".opus",
}


def download_audio_from_url(url: str, dest_path: Optional[str] = None) -> Tuple[str, str]:
    """Download an audio file from a public HTTP/HTTPS URL into a temporary file.

    Args:
        url: Direct link to the audio file.
        dest_path: Optional explicit file path to write to.

    Returns:
        tuple of (saved_file_path, original_filename)
    """
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError(f"Invalid URL protocol '{parsed.scheme}'. Only http:// and https:// URLs are supported.")

    # Determine fallback filename from URL path
    raw_path = urllib.parse.unquote(parsed.path)
    base_name = os.path.basename(raw_path).strip()
    detected_name = base_name if base_name else "audio_download.mp3"

    # Try downloading with httpx if available, fallback to urllib
    try:
        import httpx

        with httpx.Client(timeout=120.0, follow_redirects=True) as client:
            with client.stream("GET", url) as response:
                if response.status_code != 200:
                    raise RuntimeError(f"Failed to download audio from URL: HTTP status {response.status_code}")

                # Check Content-Disposition header for filename
                content_disp = response.headers.get("content-disposition", "")
                filename_match = re.search(r'filename\*?=(?:UTF-8\'\')?["\']?([^"\';\n]+)["\']?', content_disp, re.IGNORECASE)
                if filename_match:
                    detected_name = urllib.parse.unquote(filename_match.group(1).strip())

                # Check Content-Type header if extension is missing
                content_type = response.headers.get("content-type", "").split(";")[0].strip().lower()
                ext = os.path.splitext(detected_name)[1].lower()
                if not ext and content_type in MIME_TO_EXTENSION:
                    detected_name += MIME_TO_EXTENSION[content_type]
                    ext = MIME_TO_EXTENSION[content_type]

                suffix = ext if ext else ".mp3"
                if dest_path:
                    target_file = dest_path
                    with open(target_file, "wb") as f:
                        for chunk in response.iter_bytes(chunk_size=65536):
                            f.write(chunk)
                else:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                        target_file = tmp.name
                        for chunk in response.iter_bytes(chunk_size=65536):
                            tmp.write(chunk)

                return target_file, detected_name

    except ImportError:
        logger.info("'httpx' not installed, using standard library 'urllib.request'...")
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "AudioTranscriptionService/1.0"},
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            content_disp = resp.headers.get("Content-Disposition", "")
            filename_match = re.search(r'filename\*?=(?:UTF-8\'\')?["\']?([^"\';\n]+)["\']?', content_disp, re.IGNORECASE)
            if filename_match:
                detected_name = urllib.parse.unquote(filename_match.group(1).strip())

            content_type = resp.headers.get("Content-Type", "").split(";")[0].strip().lower()
            ext = os.path.splitext(detected_name)[1].lower()
            if not ext and content_type in MIME_TO_EXTENSION:
                detected_name += MIME_TO_EXTENSION[content_type]
                ext = MIME_TO_EXTENSION[content_type]

            suffix = ext if ext else ".mp3"
            if dest_path:
                target_file = dest_path
                with open(target_file, "wb") as f:
                    while chunk := resp.read(65536):
                        f.write(chunk)
            else:
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    target_file = tmp.name
                    while chunk := resp.read(65536):
                        tmp.write(chunk)

            return target_file, detected_name


def get_faster_whisper_model(model_size: str = DEFAULT_MODEL_SIZE, device: str = DEFAULT_DEVICE):
    """Retrieve or load a faster-whisper WhisperModel instance."""
    from faster_whisper import WhisperModel

    cache_key = f"faster_whisper_{model_size}_{device}"
    if cache_key not in _LOADED_MODELS:
        logger.info("Loading faster-whisper model '%s' on %s (compute_type=%s)...", model_size, device, DEFAULT_COMPUTE_TYPE)
        _LOADED_MODELS[cache_key] = WhisperModel(
            model_size,
            device=device,
            compute_type=DEFAULT_COMPUTE_TYPE,
        )
    return _LOADED_MODELS[cache_key]


def get_openai_whisper_model(model_size: str = DEFAULT_MODEL_SIZE):
    """Retrieve or load an openai-whisper model instance."""
    import whisper

    cache_key = f"openai_whisper_{model_size}"
    if cache_key not in _LOADED_MODELS:
        logger.info("Loading openai-whisper model '%s'...", model_size)
        _LOADED_MODELS[cache_key] = whisper.load_model(model_size)
    return _LOADED_MODELS[cache_key]


def transcribe_audio(
    file_path: str,
    model_size: Optional[str] = None,
    language: Optional[str] = None,
    device: Optional[str] = None,
) -> Dict[str, Any]:
    """Transcribe an audio file to text.

    Args:
        file_path: Path to the audio file or a direct HTTP/HTTPS URL.
        model_size: Whisper model size (tiny, base, small, medium, large-v3). Defaults to env or 'medium'.
        language: Optional 2-letter language code (e.g. 'en', 'es', 'fr'). If None, auto-detected.
        device: 'cpu' or 'cuda'. Defaults to 'cpu'.

    Returns:
        dict containing 'text', 'language', 'language_probability', 'duration', and 'segments'.
    """
    downloaded_temp_file = None
    actual_path = file_path

    # Check if input is a web URL
    if file_path.startswith(("http://", "https://")):
        logger.info("Downloading audio from URL: %s", file_path)
        actual_path, _ = download_audio_from_url(file_path)
        downloaded_temp_file = actual_path

    try:
        if not os.path.exists(actual_path):
            raise FileNotFoundError(f"Audio file not found: {actual_path}")

        model_name = model_size or DEFAULT_MODEL_SIZE
        dev = device or DEFAULT_DEVICE

        # Try faster-whisper first (fastest, bundled audio decoders)
        try:
            model = get_faster_whisper_model(model_size=model_name, device=dev)
            segments, info = model.transcribe(
                actual_path,
                language=language,
                beam_size=5,
                vad_filter=True,  # Filter out silence/background noise
            )

            segment_list: List[Dict[str, Any]] = []
            full_text_parts: List[str] = []

            for idx, segment in enumerate(segments):
                text_strip = segment.text.strip()
                if text_strip:
                    full_text_parts.append(text_strip)
                segment_list.append({
                    "id": idx,
                    "start": round(segment.start, 2),
                    "end": round(segment.end, 2),
                    "text": text_strip,
                })

            return {
                "text": " ".join(full_text_parts).strip(),
                "language": info.language,
                "language_probability": round(info.language_probability, 4) if info.language_probability else None,
                "duration": round(info.duration, 2) if info.duration else None,
                "segments": segment_list,
                "engine": "faster-whisper",
                "model": model_name,
            }

        except ImportError:
            logger.warning("'faster-whisper' not installed. Attempting fallback to 'openai-whisper'...")
        except Exception as exc:
            logger.warning("Error with faster-whisper (%s). Attempting fallback to openai-whisper...", exc)

        # Fallback to openai-whisper
        try:
            model = get_openai_whisper_model(model_size=model_name)
            transcribe_args: Dict[str, Any] = {}
            if language:
                transcribe_args["language"] = language

            result = model.transcribe(actual_path, **transcribe_args)

            segment_list = [
                {
                    "id": seg["id"],
                    "start": round(seg["start"], 2),
                    "end": round(seg["end"], 2),
                    "text": seg["text"].strip(),
                }
                for seg in result.get("segments", [])
            ]

            return {
                "text": result.get("text", "").strip(),
                "language": result.get("language"),
                "language_probability": None,
                "duration": None,
                "segments": segment_list,
                "engine": "openai-whisper",
                "model": model_name,
            }
        except ImportError:
            raise RuntimeError(
                "No transcription library installed! Please install faster-whisper via:\n"
                "    pip install faster-whisper\n"
                "Or install openai-whisper via:\n"
                "    pip install openai-whisper"
            ) from None

    finally:
        if downloaded_temp_file and os.path.exists(downloaded_temp_file):
            try:
                os.remove(downloaded_temp_file)
            except OSError:
                pass


def main():
    """CLI entrypoint for standalone script usage."""
    parser = argparse.ArgumentParser(
        description="Convert an audio file or audio URL to text using Whisper.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("audio_source", help="Path to local audio file or public audio URL (http/https)")
    parser.add_argument("--model", "-m", default=DEFAULT_MODEL_SIZE, help="Model size (tiny, base, small, medium, large-v3)")
    parser.add_argument("--language", "-l", default=None, help="Audio language code (e.g. en, es, fr). Auto-detect if omitted")
    parser.add_argument("--output", "-o", default=None, help="Optional output text file path to save transcription")
    parser.add_argument("--show-segments", action="store_true", help="Print timestamped segments in output")

    args = parser.parse_args()

    print(f"\n[Transcribing] {args.audio_source} using model '{args.model}'...")
    start = time.time()
    try:
        res = transcribe_audio(
            file_path=args.audio_source,
            model_size=args.model,
            language=args.language,
        )
    except Exception as e:
        print(f"\n[Error] Failed to transcribe: {e}", file=sys.stderr)
        sys.exit(1)

    elapsed = round(time.time() - start, 2)
    print(f"\n[Done in {elapsed}s]")
    if res.get("language"):
        prob_str = f" ({res['language_probability'] * 100:.1f}%)" if res.get("language_probability") else ""
        print(f"Detected Language: {res['language']}{prob_str}")

    print("\n--- Transcription Text ---")
    print(res["text"])
    print("---------------------------\n")

    if args.show_segments and res.get("segments"):
        print("--- Segments ---")
        for seg in res["segments"]:
            print(f"[{seg['start']:6.2f}s -> {seg['end']:6.2f}s] {seg['text']}")
        print("----------------\n")

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(res["text"] + "\n")
        print(f"Transcription saved to: {args.output}")


if __name__ == "__main__":
    main()
