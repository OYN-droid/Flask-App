import logging
import os
import re
import sqlite3

from flask import current_app, flash, redirect, render_template, request, send_file, url_for
from werkzeug.exceptions import RequestEntityTooLarge

import config
from services import audio, pdf_processing, vector_store


logger = logging.getLogger(__name__)


def render_index(
    current_doc=None,
    read_text=None,
    question=None,
    answer=None,
    sources=None,
    document_summary=None,
):
    """Render the index page with its complete shared template context."""
    current_document = (
        vector_store.get_document_metadata(current_doc) if current_doc else None
    )
    if current_document and document_summary is None:
        document_summary = vector_store.get_cached_summary(current_doc)
    return render_template(
        "index.html",
        documents=vector_store.list_documents(),
        current_doc=current_doc if current_document else None,
        current_document=current_document,
        read_text=read_text,
        question=question,
        answer=answer,
        sources=sources or [],
        document_summary=document_summary,
        openai_key_available=config.is_openai_key_available(),
    )


def summarize_source_documents(source_documents, default_document=None):
    """Build one concise source summary per document and page."""
    sources = []
    seen_sources = set()

    for source_document in source_documents or []:
        metadata = source_document.metadata or {}
        page = metadata.get("page")
        document_name = (
            metadata.get("original_filename")
            or default_document
            or metadata.get("collection_name")
            or "Unknown document"
        )
        source_key = (document_name, page)
        if page is None or source_key in seen_sources:
            continue
        seen_sources.add(source_key)

        is_image = metadata.get("type") == "image"
        content = source_document.page_content or ""
        content = re.sub(
            rf"^\s*Page\s+{re.escape(str(page))}(?:\s+Image\s+\d+)?:\s*",
            "",
            content,
            count=1,
            flags=re.IGNORECASE,
        )
        content = " ".join(content.split())
        if len(content) > 200:
            content = f"{content[:200].rstrip()}…"
        if is_image:
            content = f"Image description: {content}"

        sources.append(
            {
                "document": document_name,
                "page": page,
                "excerpt": content,
                "is_image": is_image,
            }
        )

    return sources


def index():
    current_doc = request.args.get("doc")
    return render_index(current_doc=current_doc)


def handle_request_entity_too_large(_error):
    max_upload_mb = current_app.config["MAX_UPLOAD_MB"]
    flash(
        f"That PDF is too large (max {max_upload_mb} MB). "
        "Try a smaller file or one with fewer/lower-resolution images."
    )
    return redirect(url_for("index"))


def delete_document():
    documentation = request.form.get("document_name")
    if not documentation:
        flash("No document selected.")
        return redirect(url_for("index"))

    document = vector_store.get_document_metadata(documentation)
    if not document:
        flash("The selected document is not available.")
        return redirect(url_for("index"))

    try:
        vector_store.chroma_client.delete_collection(name=documentation)
        removed_file = vector_store.find_uploaded_file(documentation)
        removed_uploaded_file = bool(removed_file and os.path.exists(removed_file))
        if removed_uploaded_file:
            os.remove(removed_file)

        audio_path = os.path.join(config.AUDIO_FOLDER, f"{documentation}.mp3")
        # Remove cached audio so a later upload with the same name cannot serve stale content.
        if os.path.exists(audio_path):
            os.remove(audio_path)
        vector_store.delete_document_metadata(documentation)

        if removed_uploaded_file:
            flash(f"Deleted document {document['original_filename']}.")
        else:
            flash(
                f"Deleted document {document['original_filename']}. "
                "No matching uploaded file was found."
            )
    except Exception:
        logger.exception("Failed to delete document %s", documentation)
        flash("Failed to delete the document. Please try again.")

    return redirect(url_for("index"))


def listen():
    """Redirect form submissions to the browser-loadable audio URL."""
    documentation = request.form.get("document_name")
    if not documentation:
        flash("No document selected.")
        return redirect(url_for("index"))
    return redirect(url_for("listen_document", document_name=documentation))


def listen_document(document_name):
    """Generate and stream audio for a PDF document."""
    documentation = document_name
    if not config.is_openai_key_available():
        flash("OpenAI API key is missing. Audio transcription is not available.")
        return redirect(url_for("index", doc=documentation))

    document = vector_store.get_document_metadata(documentation)
    if not document:
        flash("The selected document is not available.")
        return redirect(url_for("index"))

    audio_path = os.path.join(config.AUDIO_FOLDER, f"{documentation}.mp3")
    if not os.path.exists(audio_path):
        content = vector_store.get_document_text(documentation)
        if not content:
            flash("Could not retrieve document content for audio generation.")
            return redirect(url_for("index", doc=documentation))

        if not audio.generate_audio_from_text(content, audio_path):
            flash("Failed to generate audio. Please try again.")
            return redirect(url_for("index", doc=documentation))

    response = send_file(
        audio_path,
        mimetype="audio/mpeg",
        as_attachment=False,
        download_name=f"{os.path.splitext(document['original_filename'])[0]}.mp3",
        conditional=True,
    )
    response.headers["Accept-Ranges"] = "bytes"
    return response


def open_document():
    documentation = request.args.get("doc")
    if not documentation:
        flash("No document selected to open.")
        return redirect(url_for("index"))

    document = vector_store.get_document_metadata(documentation)
    if not document:
        flash("The selected document is not available.")
        return redirect(url_for("index"))

    return render_template(
        "view_pdf.html",
        document_name=document["original_filename"],
        pdf_url=url_for("serve_pdf", document_name=documentation),
    )


def serve_pdf(document_name):
    document = vector_store.get_document_metadata(document_name)
    if not document:
        flash("The selected document is not available.")
        return redirect(url_for("index"))

    pdf_path = vector_store.find_uploaded_file(document_name)
    if not pdf_path or not os.path.exists(pdf_path):
        flash("Uploaded PDF file could not be found.")
        return redirect(url_for("index"))

    return send_file(
        pdf_path,
        mimetype="application/pdf",
        as_attachment=False,
        download_name=document["original_filename"],
        conditional=True,
    )


def read_document():
    documentation = request.args.get("doc")
    if not documentation:
        flash("No document selected to read.")
        return redirect(url_for("index"))

    if documentation not in vector_store.list_collections():
        flash("The selected document is not available.")
        return redirect(url_for("index"))

    pdf_path = vector_store.find_uploaded_file(documentation)
    if not pdf_path or not os.path.exists(pdf_path):
        flash("Uploaded PDF file could not be found.")
        return redirect(url_for("index"))

    read_text = pdf_processing.extract_pdf_text_for_reading(pdf_path)
    return render_index(current_doc=documentation, read_text=read_text)


def summarize_document(document_name):
    document = vector_store.get_document_metadata(document_name)
    if not document:
        flash("The selected document is not available.")
        return redirect(url_for("index"))

    cached_summary = vector_store.get_cached_summary(document_name)
    if not cached_summary and not config.is_openai_key_available():
        flash(
            "OpenAI API key is missing. Set OPENAI_API_KEY or OPENAI_ADMIN_KEY and restart the app."
        )
        return redirect(url_for("index", doc=document_name))

    try:
        document_summary = cached_summary or vector_store.summarize_document(document_name)
    except Exception:
        logger.exception("Failed to summarize document %s", document_name)
        flash("Failed to summarize the document. Please try again.")
        return redirect(url_for("index", doc=document_name))

    return render_index(
        current_doc=document_name,
        document_summary=document_summary,
    )


def upload():
    if "pdf_file" not in request.files:
        flash("No file part in the request.")
        return redirect(url_for("index"))
    if not config.is_openai_key_available():
        flash(
            "OpenAI API key is missing. Set OPENAI_API_KEY or OPENAI_ADMIN_KEY and restart the app."
        )
        return redirect(url_for("index"))

    file = request.files["pdf_file"]
    if file.filename == "":
        flash("No file selected.")
        return redirect(url_for("index"))
    if not config.allowed_file(file.filename):
        flash("Only PDF files are allowed.")
        return redirect(url_for("index"))

    original_filename = os.path.basename(file.filename.replace("\\", "/"))
    document = None
    save_path = None
    try:
        document = vector_store.reserve_document_metadata(original_filename)
        collection_name = document["collection_name"]
        save_path = os.path.join(config.UPLOAD_FOLDER, document["stored_filename"])
        file.save(save_path)
        vector_store.build_vector_store(save_path, collection_name)
    except Exception:
        logger.exception("Failed to process uploaded PDF %s", original_filename)
        try:
            if save_path and os.path.exists(save_path):
                os.remove(save_path)
        except OSError:
            logger.exception("Failed to remove incomplete upload %s", save_path)
        if document:
            try:
                collection_names = {
                    collection.name
                    for collection in vector_store.chroma_client.list_collections()
                }
                if document["collection_name"] in collection_names:
                    vector_store.chroma_client.delete_collection(
                        name=document["collection_name"]
                    )
            except Exception:
                logger.exception(
                    "Failed to remove incomplete collection %s",
                    document["collection_name"],
                )
            try:
                vector_store.delete_document_metadata(document["collection_name"])
            except sqlite3.Error:
                logger.exception(
                    "Failed to remove incomplete document metadata %s",
                    document["collection_name"],
                )
        flash("Failed to process the PDF. Please try again or use a different file.")
        return redirect(url_for("index"))

    flash(f"Uploaded and indexed {original_filename}.")
    return redirect(url_for("index", doc=collection_name))


def ask():
    documentation = request.form.get("document_name")
    question = request.form.get("question")

    if not documentation:
        flash("No document selected.")
        return redirect(url_for("index"))
    if not question:
        flash("Please enter a question.")
        return redirect(url_for("index", doc=documentation))
    if not config.is_openai_key_available():
        flash(
            "OpenAI API key is missing. Set OPENAI_API_KEY or OPENAI_ADMIN_KEY and restart the app."
        )
        return redirect(url_for("index", doc=documentation))
    if documentation not in vector_store.list_collections():
        flash("The selected document is not available. Upload the PDF again.")
        return redirect(url_for("index"))

    try:
        qa_chain = vector_store.get_qa_chain(documentation)
        result = qa_chain.invoke({"query": question})
        answer = result["result"]
        document = vector_store.get_document_metadata(documentation)
        sources = summarize_source_documents(
            result.get("source_documents", []),
            default_document=document["original_filename"],
        )
    except Exception:
        logger.exception("Failed to answer question for document %s", documentation)
        flash("Failed to answer the question. Please try again.")
        return redirect(url_for("index", doc=documentation))

    return render_index(
        current_doc=documentation,
        question=question,
        answer=answer,
        sources=sources,
    )


def ask_multi():
    collection_names = list(dict.fromkeys(request.form.getlist("document_names")))
    question = request.form.get("question")

    if len(collection_names) < 2:
        flash("Select at least two documents to ask across.")
        return redirect(url_for("index"))
    if not question:
        flash("Please enter a question.")
        return redirect(url_for("index"))
    if not config.is_openai_key_available():
        flash(
            "OpenAI API key is missing. Set OPENAI_API_KEY or OPENAI_ADMIN_KEY and restart the app."
        )
        return redirect(url_for("index"))

    available_collections = set(vector_store.list_collections())
    if any(name not in available_collections for name in collection_names):
        flash("One or more selected documents are not available. Please select them again.")
        return redirect(url_for("index"))

    try:
        result = vector_store.answer_across_documents(collection_names, question)
        answer = result["result"]
        sources = summarize_source_documents(result.get("source_documents", []))
    except Exception:
        logger.exception("Failed to answer question across selected documents")
        flash("Failed to answer the question. Please try again.")
        return redirect(url_for("index"))

    return render_index(question=question, answer=answer, sources=sources)


def register_routes(app):
    """Register routes while preserving the app's existing endpoint names and paths."""
    app.add_url_rule("/", endpoint="index", view_func=index, methods=["GET"])
    app.add_url_rule(
        "/delete", endpoint="delete_document", view_func=delete_document, methods=["POST"]
    )
    app.add_url_rule("/listen", endpoint="listen", view_func=listen, methods=["POST"])
    app.add_url_rule(
        "/listen/<document_name>",
        endpoint="listen_document",
        view_func=listen_document,
        methods=["GET"],
    )
    app.add_url_rule(
        "/open", endpoint="open_document", view_func=open_document, methods=["GET"]
    )
    app.add_url_rule(
        "/pdf/<document_name>",
        endpoint="serve_pdf",
        view_func=serve_pdf,
        methods=["GET"],
    )
    app.add_url_rule(
        "/read", endpoint="read_document", view_func=read_document, methods=["GET"]
    )
    app.add_url_rule(
        "/summarize/<document_name>",
        endpoint="summarize_document",
        view_func=summarize_document,
        methods=["GET"],
    )
    app.add_url_rule("/upload", endpoint="upload", view_func=upload, methods=["POST"])
    app.add_url_rule("/ask", endpoint="ask", view_func=ask, methods=["POST"])
    app.add_url_rule(
        "/ask-multi", endpoint="ask_multi", view_func=ask_multi, methods=["POST"]
    )
    app.register_error_handler(RequestEntityTooLarge, handle_request_entity_too_large)
