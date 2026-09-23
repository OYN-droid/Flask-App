# PDF Reader & Analyzer

A Flask app that turns uploaded PDFs into something you can search, question, and listen to — not just view. It extracts text and embedded images, indexes both into a per-document vector store, and answers questions about the content with page-level citations, using OpenAI models for embeddings, vision, and chat.

Built as a portfolio project to explore retrieval-augmented generation (RAG), multimodal document understanding, and running an LLM-backed app with real error handling, tests, and a modular codebase — not just a notebook demo.

## Features

- **Upload and index PDFs** — text is extracted per page (via `pdfplumber`) and embedded images are described and classified (photo, chart, table, or diagram) using a vision-capable OpenAI model, with Tesseract OCR as a supplementary text hint. Everything is chunked and embedded into a per-document [Chroma](https://www.trychroma.com/) vector store.
- **Ask questions, get cited answers** — questions are answered using retrieval-augmented generation over the indexed document, with the response showing which page(s) it drew from and an expandable excerpt for each citation. Citations from chart/table images are labeled distinctly from body text.
- **Ask across multiple documents** — select two or more indexed documents and ask a single question that's answered by retrieving and ranking context across all of them, with citations naming both the source file and page.
- **Chart and table-aware image analysis** — embedded charts are described with their approximate axes, series, and data trend; tables are extracted as Markdown tables, rather than a generic one-line image caption.
- **Document summaries** — generate and cache an executive summary and key-points list for a document using a map-reduce summarization chain over its indexed content.
- **Read, view, and listen** — read the extracted text inline, view the original PDF in an embedded viewer, or listen to a narrated audio version generated via OpenAI's TTS API (chunked and concatenated to cover the full document, then cached).
- **Collision-safe storage** — uploaded documents are tracked in a small SQLite metadata store mapping internal collection IDs to original filenames, so two different files that sanitize to a similar name don't silently overwrite each other's index.
- **Hardened by default** — debug mode and public network binding are off unless explicitly enabled via environment variables, a secret key is required outside local dev, uploads over a configurable size limit get a friendly error instead of a raw server error page, and errors are logged server-side without leaking internal details to the browser.

## Tech stack

Flask · LangChain (`langchain-community`, `langchain-openai`) · ChromaDB · OpenAI (embeddings, `gpt-4o-mini` for vision/QA, `tts-1` for narration) · `pdfplumber` · Tesseract OCR (via `pytesseract`) · SQLite · pytest

## Project structure

```
app.py                       # App factory / startup
config.py                    # Environment variables, paths, constants
routes.py                    # Flask routes
services/
  pdf_processing.py          # Text and image extraction, vision-based image analysis
  vector_store.py            # Chroma, embeddings, QA chain, summaries, document metadata
  audio.py                   # TTS generation and chunking
templates/
  index.html                 # Main UI (upload, documents, ask, summaries)
  view_pdf.html               # Inline PDF viewer
tests/
  test_app.py                 # Route and helper tests (mocked OpenAI/Chroma calls)
```

## Setup

**Python version:** this project currently requires **Python 3.11**. `chromadb`'s `onnxruntime` dependency does not yet have a compatible build for Python 3.14, so a newer default Python will fail during `pip install`.

1. Create and activate a virtual environment (with Python 3.11):
   ```bash
   python3.11 -m venv .venv
   source .venv/bin/activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Install Tesseract OCR (used as a supplementary hint for image analysis — the app still works without it, with reduced accuracy on image-heavy pages):
   ```bash
   # macOS
   brew install tesseract

   # Debian/Ubuntu
   sudo apt-get install tesseract-ocr
   ```

4. Create a `.env` file in the project root (see `.env.example` for the full list):
   ```bash
   OPENAI_API_KEY=your_openai_api_key_here
   FLASK_SECRET_KEY=$(python -c "import secrets; print(secrets.token_hex(32))")
   ```
   `FLASK_SECRET_KEY` is required unless you set `FLASK_DEBUG=1` or `FLASK_ENV=development`, in which case a throwaway key is generated automatically for local dev.

5. Run the app:
   ```bash
   python app.py
   ```
   By default this binds to `127.0.0.1:5000` with debug mode off. Set `FLASK_DEBUG=1` for local development (auto-reload, debugger), and `FLASK_RUN_HOST` if you need to bind elsewhere — do not enable debug mode on a network-reachable host.

6. Open `http://127.0.0.1:5000` in your browser.

### Configuration

| Variable | Default | Purpose |
|---|---|---|
| `OPENAI_API_KEY` / `OPENAI_ADMIN_KEY` | — | Required for indexing, Q&A, image analysis, and audio narration. |
| `FLASK_SECRET_KEY` | — (required outside dev mode) | Signs sessions and flash messages. |
| `FLASK_DEBUG` / `FLASK_ENV` | off | Enables debug mode and an auto-generated dev secret key. |
| `FLASK_RUN_HOST` | `127.0.0.1` | Host to bind to. |
| `MAX_UPLOAD_MB` | `50` | Maximum accepted PDF upload size. |

## Running tests

```bash
pip install -r requirements-dev.txt
.venv/bin/python -m pytest
```

Use the module invocation (`python -m pytest`) rather than a bare `pytest` command — the direct launcher can hit an import-path collection error depending on how the virtualenv is set up.

## Known limitations

- Chart and table data extracted from images is the vision model's best reading of the picture, not a precise data extraction — treat it as an approximation, not ground truth.
- Indexing and summarizing large, image-heavy PDFs happens synchronously within the request; very large documents can take a while to process.
- This is a single-user local app with no authentication — it's not intended to be deployed publicly without adding access control and rate limiting.

## License

MIT — see [LICENSE](LICENSE).