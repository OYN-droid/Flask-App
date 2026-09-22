import logging
import os
import secrets

from flask import Flask


logging.basicConfig(level=logging.INFO)

import config
from routes import register_routes
from services import vector_store


def create_app():
    config.ensure_storage_directories()

    flask_app = Flask(__name__)
    secret_key = os.environ.get("FLASK_SECRET_KEY")
    if not secret_key:
        if not config.is_local_development():
            raise RuntimeError(
                "FLASK_SECRET_KEY is required. Set it to a long random value before starting the app."
            )
        secret_key = secrets.token_hex(32)

    flask_app.secret_key = secret_key
    flask_app.config["MAX_UPLOAD_MB"] = config.get_max_upload_mb()
    flask_app.config["MAX_CONTENT_LENGTH"] = (
        flask_app.config["MAX_UPLOAD_MB"] * 1024 * 1024
    )

    vector_store.initialize_metadata_store()
    vector_store.backfill_legacy_document_metadata()
    register_routes(flask_app)
    return flask_app


app = create_app()


if __name__ == "__main__":
    app.run(
        debug=config.is_local_development(),
        host=os.environ.get("FLASK_RUN_HOST", "127.0.0.1"),
        port=5000,
    )
