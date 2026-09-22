import os

from dotenv import load_dotenv


BASE_DIR = os.path.abspath(os.path.dirname(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
PERSIST_DIR = os.path.join(BASE_DIR, "db")
AUDIO_FOLDER = os.path.join(BASE_DIR, "audio")
METADATA_DB_PATH = os.path.join(PERSIST_DIR, "documents.sqlite3")
ALLOWED_EXTENSIONS = {"pdf"}
DEFAULT_MAX_UPLOAD_MB = 50
TTS_MAX_CHARS = 4096

# Environment values must be loaded before app startup validates its configuration.
load_dotenv(os.path.join(BASE_DIR, ".env"))


def ensure_storage_directories():
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(PERSIST_DIR, exist_ok=True)
    os.makedirs(AUDIO_FOLDER, exist_ok=True)


def get_max_upload_mb():
    value = os.environ.get("MAX_UPLOAD_MB", str(DEFAULT_MAX_UPLOAD_MB))
    try:
        max_upload_mb = int(value)
    except ValueError:
        raise RuntimeError("MAX_UPLOAD_MB must be a positive whole number.") from None
    if max_upload_mb <= 0:
        raise RuntimeError("MAX_UPLOAD_MB must be a positive whole number.")
    return max_upload_mb


def is_local_development():
    """Return whether an explicit environment setting enables local development."""
    return os.environ.get("FLASK_DEBUG", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    } or os.environ.get("FLASK_ENV", "").strip().lower() == "development"


def is_openai_key_available():
    """Return True if an OpenAI API key is present in the environment."""
    return bool(os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENAI_ADMIN_KEY"))


def get_openai_api_key():
    key = os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENAI_ADMIN_KEY")
    if not key:
        raise RuntimeError(
            "OpenAI API key is not set. Set OPENAI_API_KEY or OPENAI_ADMIN_KEY in your environment."
        )
    return key


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
