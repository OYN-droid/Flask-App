import logging
import os
import re
import shutil
import tempfile

from openai import OpenAI as OpenAIClient

import config


logger = logging.getLogger(__name__)


def split_text_for_tts(text, max_chars=config.TTS_MAX_CHARS):
    """Split text within the TTS limit, preferring paragraph and sentence breaks."""
    chunks = []
    remaining = text.strip()

    while remaining:
        if len(remaining) <= max_chars:
            chunks.append(remaining)
            break

        window = remaining[:max_chars]
        minimum_break = max_chars // 2
        break_at = window.rfind("\n\n")

        if break_at < minimum_break:
            sentence_breaks = list(re.finditer(r'[.!?]["\']?\s+', window))
            break_at = sentence_breaks[-1].end() if sentence_breaks else -1

        if break_at < minimum_break:
            break_at = window.rfind(" ") + 1

        if break_at <= 0:
            break_at = max_chars

        chunks.append(remaining[:break_at].strip())
        remaining = remaining[break_at:].lstrip()

    return chunks


def generate_audio_from_text(text, audio_path):
    """Generate and combine TTS audio segments for the complete supplied text."""
    combined_path = None
    try:
        client = OpenAIClient(api_key=config.get_openai_api_key())
        chunks = split_text_for_tts(text)
        if not chunks:
            return False

        audio_dir = os.path.dirname(os.path.abspath(audio_path))
        with tempfile.TemporaryDirectory(prefix="tts-segments-", dir=audio_dir) as temp_dir:
            segment_paths = []
            for index, chunk in enumerate(chunks):
                response = client.audio.speech.create(
                    model="tts-1",
                    voice="alloy",
                    input=chunk,
                )
                segment_path = os.path.join(temp_dir, f"segment-{index:04d}.mp3")
                response.stream_to_file(segment_path)
                segment_paths.append(segment_path)

            with tempfile.NamedTemporaryFile(
                mode="wb",
                prefix="tts-combined-",
                suffix=".mp3",
                dir=audio_dir,
                delete=False,
            ) as combined_file:
                combined_path = combined_file.name
                for segment_path in segment_paths:
                    with open(segment_path, "rb") as segment_file:
                        shutil.copyfileobj(segment_file, combined_file)

            os.replace(combined_path, audio_path)
            combined_path = None
        return True
    except Exception:
        logger.exception("Failed to generate audio")
        return False
    finally:
        if combined_path and os.path.exists(combined_path):
            os.remove(combined_path)
