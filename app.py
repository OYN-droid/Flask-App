import os
import re
import base64
import io
from dotenv import load_dotenv
from flask import Flask, request, render_template, redirect, url_for, flash, send_file
from werkzeug.utils import secure_filename

# Updated LangChain imports (langchain-community + langchain-openai packages required)
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_community.llms import OpenAI
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.schema import Document

# Updated ChromaDB import (v0.4+ API)
import chromadb
import pdfplumber
from openai import OpenAI as OpenAIClient

# OCR dependencies
from PIL import Image
import pytesseract

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
PERSIST_DIR = os.path.join(BASE_DIR, "db")
AUDIO_FOLDER = os.path.join(BASE_DIR, "audio")
ALLOWED_EXTENSIONS = {"pdf"}
MAX_CONTENT_LENGTH = 32 * 1024 * 1024  # 32 MB upload limit

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PERSIST_DIR, exist_ok=True)
os.makedirs(AUDIO_FOLDER, exist_ok=True)

# Load environment variables BEFORE checking for API key
load_dotenv(os.path.join(BASE_DIR, ".env"))

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "change-me")
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH

# FIX 1: Use PersistentClient instead of deprecated Client(Settings(chroma_db_impl=...))
chroma_client = chromadb.PersistentClient(path=PERSIST_DIR)

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENAI_ADMIN_KEY")


def is_openai_key_available():
    """Return True if an OpenAI API key is present in the environment."""
    return bool(os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENAI_ADMIN_KEY"))


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def safe_collection_name(filename):
    name = os.path.splitext(filename)[0]
    return re.sub(r"[^a-zA-Z0-9_-]", "_", name)


def list_collections():
    return [collection.name for collection in chroma_client.list_collections()]


def find_uploaded_file(collection_name):
    for filename in os.listdir(UPLOAD_FOLDER):
        if safe_collection_name(filename) == collection_name:
            return os.path.join(UPLOAD_FOLDER, filename)
    return None


def generate_audio_from_text(text, audio_path):
    """Generate audio file from text using OpenAI's TTS API."""
    try:
        api_key = get_openai_api_key()
        client = OpenAIClient(api_key=api_key)
        response = client.audio.speech.create(
            model="tts-1",
            voice="alloy",
            input=text[:4096],  # TTS has a 4096 character limit per request
        )
        response.stream_to_file(audio_path)
        return True
    except Exception as e:
        print(f"Failed to generate audio: {e}")
        return False


def get_document_text(collection_name):
    """Retrieve all text content from a document (text + image descriptions)."""
    try:
        retriever = Chroma(
            persist_directory=PERSIST_DIR,
            embedding_function=OpenAIEmbeddings(openai_api_key=get_openai_api_key()),
            collection_name=collection_name,
        ).as_retriever(search_kwargs={"k": 100})
        docs = retriever.get_relevant_documents("")
        content = "\n\n".join([doc.page_content for doc in docs])
        return content
    except Exception as e:
        print(f"Failed to retrieve document text: {e}")
        return ""


def extract_pdf_text_for_reading(pdf_path):
    """Extract readable text from a PDF for inline reading."""
    try:
        pages = []
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                text = page.extract_text() or ""
                if text.strip():
                    pages.append(f"Page {page_num}:\n{text}")
        if not pages:
            return "No readable text was found in this PDF."
        return "\n\n".join(pages)
    except Exception as e:
        return f"Failed to read PDF text: {e}"


def describe_image_with_vision(image_bytes):
    """Describe an image using local OCR (pytesseract) and a text LLM.

    Steps:
    1. Run OCR on the image to extract any visible text.
    2. If OCR produced text, prompt the text model to summarize and infer visual details.
    3. If OCR produced no text, ask the model to describe what can be inferred from the image bytes' context.
    """
    try:
        # Run OCR
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        ocr_text = pytesseract.image_to_string(img).strip()

        prompt_parts = []
        if ocr_text:
            prompt_parts.append("I extracted the following text from an image via OCR:\n\n" + ocr_text)
            prompt_parts.append(
                "Please provide a concise description of the image, highlight any important text, and infer likely visual elements (layout, headings, tables, logos)."
            )
        else:
            prompt_parts.append(
                "I could not extract readable text from this image. Please provide a concise description of visual elements you can infer (colors, objects, likely content)."
            )

        prompt = "\n\n".join(prompt_parts)

        # Call text LLM to generate a human-friendly description
        client = OpenAIClient(api_key=get_openai_api_key())
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500,
        )
        # Response object shape may vary; attempt to return text safely
        try:
            return response.choices[0].message.content
        except Exception:
            # Fallback to raw content attribute if present
            return str(response)
    except Exception as e:
        return f"[Image description unavailable: {e}]"


def extract_text_and_images_from_pdf(pdf_path):
    """Extract text and image descriptions from a PDF."""
    documents = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                # Extract text
                text = page.extract_text() or ""
                if text.strip():
                    documents.append(
                        Document(
                            page_content=f"Page {page_num}:\n{text}",
                            metadata={"page": page_num, "type": "text"},
                        )
                    )

                # Extract and describe images
                if hasattr(page, "images") and page.images:
                    for img_num, img in enumerate(page.images, 1):
                        try:
                            # Get image bytes
                            img_bytes = img.get("stream").get_rawdata()
                            if img_bytes:
                                description = describe_image_with_vision(img_bytes)
                                documents.append(
                                    Document(
                                        page_content=f"Page {page_num} Image {img_num}:\n{description}",
                                        metadata={"page": page_num, "image": img_num, "type": "image"},
                                    )
                                )
                        except Exception as e:
                            print(f"Failed to process image on page {page_num}: {e}")
    except Exception as e:
        raise ValueError(f"Failed to extract content from PDF: {str(e)}")
    return documents


def build_vector_store(pdf_path, collection_name):
    if not is_openai_key_available():
        raise RuntimeError(
            "OpenAI API key is missing. Set OPENAI_API_KEY or OPENAI_ADMIN_KEY and restart the app."
        )

    if collection_name in list_collections():
        chroma_client.delete_collection(name=collection_name)

    # Extract text and images from PDF
    documents = extract_text_and_images_from_pdf(pdf_path)
    if not documents:
        raise ValueError("No text could be extracted from the uploaded PDF.")

    # Split documents into chunks
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    split_documents = splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings(openai_api_key=get_openai_api_key())

    # FIX 2: store.persist() no longer exists; PersistentClient handles it automatically
    store = Chroma.from_documents(
        split_documents,
        embeddings,
        persist_directory=PERSIST_DIR,
        collection_name=collection_name,
    )
    return store


def get_openai_api_key():
    key = os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENAI_ADMIN_KEY")
    if not key:
        raise RuntimeError(
            "OpenAI API key is not set. Set OPENAI_API_KEY or OPENAI_ADMIN_KEY in your environment."
        )
    return key


def get_qa_chain(collection_name):
    api_key = get_openai_api_key()
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)

    retriever = Chroma(
        persist_directory=PERSIST_DIR,
        embedding_function=embeddings,
        collection_name=collection_name,
    ).as_retriever(search_kwargs={"k": 4})
    llm = OpenAI(temperature=0, openai_api_key=api_key)
    return RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)


@app.route("/", methods=["GET"])
def index():
    current_doc = request.args.get("doc")
    collections = list_collections()
    return render_template(
        "index.html",
        collections=collections,
        current_doc=current_doc,
        read_text=None,
        answer=None,
        question=None,
        openai_key_available=is_openai_key_available(),
    )


@app.route("/delete", methods=["POST"])
def delete_document():
    documentation = request.form.get("document_name")
    if not documentation:
        flash("No document selected.")
        return redirect(url_for("index"))

    if documentation not in list_collections():
        flash("The selected document is not available.")
        return redirect(url_for("index"))

    try:
        chroma_client.delete_collection(name=documentation)
        removed_file = find_uploaded_file(documentation)
        if removed_file and os.path.exists(removed_file):
            os.remove(removed_file)
            flash(f"Deleted document {documentation} and uploaded file {os.path.basename(removed_file)}.")
        else:
            flash(f"Deleted document {documentation}. No matching uploaded file was found.")
    except Exception as exc:
        flash(f"Failed to delete document: {exc}")

    return redirect(url_for("index"))


@app.route("/listen", methods=["POST"])
def listen():
    """Redirect form submissions to the browser-loadable audio URL."""
    documentation = request.form.get("document_name")

    if not documentation:
        flash("No document selected.")
        return redirect(url_for("index"))

    return redirect(url_for("listen_document", document_name=documentation))


@app.route("/listen/<document_name>", methods=["GET"])
def listen_document(document_name):
    """Generate and stream audio for a PDF document."""
    documentation = document_name

    if not is_openai_key_available():
        flash("OpenAI API key is missing. Audio transcription is not available.")
        return redirect(url_for("index", doc=documentation))

    if documentation not in list_collections():
        flash("The selected document is not available.")
        return redirect(url_for("index"))

    audio_path = os.path.join(AUDIO_FOLDER, f"{documentation}.mp3")

    # Generate audio if not already cached
    if not os.path.exists(audio_path):
        content = get_document_text(documentation)
        if not content:
            flash("Could not retrieve document content for audio generation.")
            return redirect(url_for("index", doc=documentation))

        if not generate_audio_from_text(content, audio_path):
            flash("Failed to generate audio. Please try again.")
            return redirect(url_for("index", doc=documentation))

    response = send_file(
        audio_path,
        mimetype="audio/mpeg",
        as_attachment=False,
        download_name=f"{documentation}.mp3",
        conditional=True,
    )
    response.headers["Content-Disposition"] = f'inline; filename="{documentation}.mp3"'
    response.headers["Accept-Ranges"] = "bytes"
    return response

@app.route("/open", methods=["GET"])
def open_document():
    documentation = request.args.get("doc")
    if not documentation:
        flash("No document selected to open.")
        return redirect(url_for("index"))

    if documentation not in list_collections():
        flash("The selected document is not available.")
        return redirect(url_for("index"))

    return render_template(
        "view_pdf.html",
        document_name=documentation,
        pdf_url=url_for("serve_pdf", document_name=documentation),
    )


@app.route("/pdf/<document_name>", methods=["GET"])
def serve_pdf(document_name):
    if document_name not in list_collections():
        flash("The selected document is not available.")
        return redirect(url_for("index"))

    pdf_path = find_uploaded_file(document_name)
    if not pdf_path or not os.path.exists(pdf_path):
        flash("Uploaded PDF file could not be found.")
        return redirect(url_for("index"))

    response = send_file(
        pdf_path,
        mimetype="application/pdf",
        as_attachment=False,
        download_name=os.path.basename(pdf_path),
        conditional=True,
    )
    response.headers["Content-Disposition"] = f'inline; filename="{os.path.basename(pdf_path)}"'
    return response


@app.route("/read", methods=["GET"])
def read_document():
    documentation = request.args.get("doc")
    if not documentation:
        flash("No document selected to read.")
        return redirect(url_for("index"))

    if documentation not in list_collections():
        flash("The selected document is not available.")
        return redirect(url_for("index"))

    pdf_path = find_uploaded_file(documentation)
    if not pdf_path or not os.path.exists(pdf_path):
        flash("Uploaded PDF file could not be found.")
        return redirect(url_for("index"))

    read_text = extract_pdf_text_for_reading(pdf_path)
    collections = list_collections()
    return render_template(
        "index.html",
        collections=collections,
        current_doc=documentation,
        read_text=read_text,
        question=None,
        answer=None,
        openai_key_available=is_openai_key_available(),
    )


@app.route("/upload", methods=["POST"])
def upload():
    if "pdf_file" not in request.files:
        flash("No file part in the request.")
        return redirect(url_for("index"))
    if not is_openai_key_available():
        flash(
            'OpenAI API key is missing. Set OPENAI_API_KEY or OPENAI_ADMIN_KEY and restart the app.'
        )
        return redirect(url_for('index'))
    file = request.files["pdf_file"]
    if file.filename == "":
        flash("No file selected.")
        return redirect(url_for("index"))

    if not allowed_file(file.filename):
        flash("Only PDF files are allowed.")
        return redirect(url_for("index"))

    # FIX 3: Sanitize filename to prevent path traversal attacks
    filename = secure_filename(file.filename)
    save_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(save_path)

    collection_name = safe_collection_name(filename)
    try:
        build_vector_store(save_path, collection_name)
    except Exception as exc:
        flash(f"Failed to process PDF: {exc}")
        return redirect(url_for("index"))

    flash(f"Uploaded and indexed {filename}.")
    return redirect(url_for("index", doc=collection_name))


@app.route("/ask", methods=["POST"])
def ask():
    documentation = request.form.get("document_name")
    question = request.form.get("question")

    if not documentation:
        flash("No document selected.")
        return redirect(url_for("index"))

    if not question:
        flash("Please enter a question.")
        return redirect(url_for("index", doc=documentation))

    if not is_openai_key_available():
        flash(
            "OpenAI API key is missing. Set OPENAI_API_KEY or OPENAI_ADMIN_KEY and restart the app."
        )
        return redirect(url_for("index", doc=documentation))

    if documentation not in list_collections():
        flash("The selected document is not available. Upload the PDF again.")
        return redirect(url_for("index"))

    try:
        qa_chain = get_qa_chain(documentation)
        answer = qa_chain.run(question)
    except Exception as exc:
        flash(f"Failed to answer the question: {exc}")
        return redirect(url_for("index", doc=documentation))

    collections = list_collections()
    return render_template(
        "index.html",
        collections=collections,
        current_doc=documentation,
        question=question,
        answer=answer,
    )


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
