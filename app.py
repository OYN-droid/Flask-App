import logging
import os
import secrets

from flask import Flask, redirect, render_template, request, session, url_for


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
    flask_app.config["SITE_PASSWORD"] = config.get_site_password()

    @flask_app.before_request
    def require_site_password():
        site_password = flask_app.config["SITE_PASSWORD"]
        if not site_password or request.endpoint in {"login", "static"}:
            return None
        if session.get("site_authenticated") is True:
            return None
        return redirect(url_for("login"))

    @flask_app.route("/login", methods=["GET", "POST"])
    def login():
        site_password = flask_app.config["SITE_PASSWORD"]
        if not site_password:
            return redirect(url_for("index"))

        error = None
        if request.method == "POST":
            submitted_password = request.form.get("password", "")
            if secrets.compare_digest(
                submitted_password.encode("utf-8"), site_password.encode("utf-8")
            ):
                session.clear()
                session["site_authenticated"] = True
                return redirect(url_for("index"))
            error = "Incorrect password. Please try again."

        return render_template("login.html", error=error), 401 if error else 200

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
