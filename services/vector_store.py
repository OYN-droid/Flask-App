import json
import logging
import os
import re
import secrets
import sqlite3

import chromadb
from langchain_community.llms import OpenAI
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from werkzeug.utils import secure_filename

import config
from services import pdf_processing

try:
    from langchain_text_splitters import CharacterTextSplitter
except ImportError:  # LangChain < 0.3
    from langchain.text_splitter import CharacterTextSplitter

try:
    from langchain_classic.chains import RetrievalQA
    from langchain_classic.chains.summarize import load_summarize_chain
except ImportError:  # LangChain < 1.0
    from langchain.chains import RetrievalQA
    from langchain.chains.summarize import load_summarize_chain


logger = logging.getLogger(__name__)

# PersistentClient is the supported ChromaDB API for an on-disk collection store.
config.ensure_storage_directories()
chroma_client = chromadb.PersistentClient(path=config.PERSIST_DIR)


def safe_collection_name(filename):
    name = os.path.splitext(filename)[0]
    return re.sub(r"[^a-zA-Z0-9_-]", "_", name) or "document"


def metadata_connection():
    connection = sqlite3.connect(config.METADATA_DB_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def initialize_metadata_store():
    with metadata_connection() as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS documents (
                collection_name TEXT PRIMARY KEY,
                original_filename TEXT NOT NULL,
                stored_filename TEXT NOT NULL UNIQUE,
                uploaded_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS summaries (
                collection_name TEXT PRIMARY KEY,
                executive_summary TEXT NOT NULL,
                key_points TEXT NOT NULL,
                generated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (collection_name) REFERENCES documents(collection_name)
            )
            """
        )


def get_document_metadata(collection_name):
    with metadata_connection() as connection:
        row = connection.execute(
            """
            SELECT collection_name, original_filename, stored_filename, uploaded_at
            FROM documents
            WHERE collection_name = ?
            """,
            (collection_name,),
        ).fetchone()
    return dict(row) if row else None


def list_documents():
    with metadata_connection() as connection:
        rows = connection.execute(
            """
            SELECT collection_name, original_filename, stored_filename, uploaded_at
            FROM documents
            ORDER BY uploaded_at DESC, collection_name
            """
        ).fetchall()
    return [dict(row) for row in rows]


def list_collections():
    return [document["collection_name"] for document in list_documents()]


def reserve_document_metadata(original_filename):
    """Reserve unique collection and storage names for an uploaded document."""
    base_name = safe_collection_name(secure_filename(original_filename))
    existing_chroma_names = {
        collection.name for collection in chroma_client.list_collections()
    }

    while True:
        collection_name = base_name
        while (
            collection_name in existing_chroma_names
            or get_document_metadata(collection_name)
        ):
            collection_name = f"{base_name}-{secrets.token_hex(4)}"

        stored_filename = f"{collection_name}.pdf"
        try:
            with metadata_connection() as connection:
                connection.execute(
                    """
                    INSERT INTO documents (
                        collection_name, original_filename, stored_filename
                    ) VALUES (?, ?, ?)
                    """,
                    (collection_name, original_filename, stored_filename),
                )
            return get_document_metadata(collection_name)
        except sqlite3.IntegrityError:
            existing_chroma_names.add(collection_name)


def delete_document_metadata(collection_name):
    with metadata_connection() as connection:
        connection.execute(
            "DELETE FROM summaries WHERE collection_name = ?",
            (collection_name,),
        )
        connection.execute(
            "DELETE FROM documents WHERE collection_name = ?",
            (collection_name,),
        )


def get_cached_summary(collection_name):
    with metadata_connection() as connection:
        row = connection.execute(
            """
            SELECT executive_summary, key_points, generated_at
            FROM summaries
            WHERE collection_name = ?
            """,
            (collection_name,),
        ).fetchone()
    if not row:
        return None
    try:
        key_points = json.loads(row["key_points"])
    except (TypeError, json.JSONDecodeError):
        logger.error("Invalid cached summary for collection %s", collection_name)
        invalidate_cached_summary(collection_name)
        return None
    return {
        "executive_summary": row["executive_summary"],
        "key_points": key_points,
        "generated_at": row["generated_at"],
    }


def cache_summary(collection_name, executive_summary, key_points):
    with metadata_connection() as connection:
        connection.execute(
            """
            INSERT INTO summaries (collection_name, executive_summary, key_points)
            VALUES (?, ?, ?)
            ON CONFLICT(collection_name) DO UPDATE SET
                executive_summary = excluded.executive_summary,
                key_points = excluded.key_points,
                generated_at = CURRENT_TIMESTAMP
            """,
            (collection_name, executive_summary, json.dumps(key_points)),
        )


def invalidate_cached_summary(collection_name):
    with metadata_connection() as connection:
        connection.execute(
            "DELETE FROM summaries WHERE collection_name = ?",
            (collection_name,),
        )


def backfill_legacy_document_metadata():
    """Register collections created before the metadata store was introduced."""
    uploaded_files = {}
    for filename in os.listdir(config.UPLOAD_FOLDER):
        uploaded_files.setdefault(safe_collection_name(filename), filename)

    with metadata_connection() as connection:
        for collection in chroma_client.list_collections():
            existing = connection.execute(
                "SELECT 1 FROM documents WHERE collection_name = ?",
                (collection.name,),
            ).fetchone()
            if existing:
                continue

            stored_filename = uploaded_files.get(
                collection.name, f"{collection.name}.pdf"
            )
            connection.execute(
                """
                INSERT OR IGNORE INTO documents (
                    collection_name, original_filename, stored_filename
                ) VALUES (?, ?, ?)
                """,
                (collection.name, stored_filename, stored_filename),
            )


def find_uploaded_file(collection_name):
    document = get_document_metadata(collection_name)
    if not document:
        return None
    return os.path.join(config.UPLOAD_FOLDER, document["stored_filename"])


def get_document_text(collection_name):
    """Retrieve all text content from a document (text + image descriptions)."""
    try:
        collection = chroma_client.get_collection(name=collection_name)
        result = collection.get(include=["documents", "metadatas"])
        documents = result.get("documents") or []
        metadatas = result.get("metadatas") or [{} for _ in documents]

        indexed_documents = list(enumerate(zip(documents, metadatas)))

        def document_order(item):
            original_index, (_, metadata) = item
            metadata = metadata or {}
            if isinstance(metadata.get("chunk_index"), int):
                return (0, metadata["chunk_index"], original_index)
            page = metadata.get("page")
            return (1, page if isinstance(page, int) else float("inf"), original_index)

        indexed_documents.sort(key=document_order)
        return "\n\n".join(
            document
            for _, (document, _) in indexed_documents
            if document and document.strip()
        )
    except Exception:
        logger.exception("Failed to retrieve document text for collection %s", collection_name)
        return ""


def get_indexed_documents(collection_name):
    """Return all indexed chunks in their original document order."""
    collection = chroma_client.get_collection(name=collection_name)
    result = collection.get(include=["documents", "metadatas"])
    documents = result.get("documents") or []
    metadatas = result.get("metadatas") or [{} for _ in documents]
    indexed_documents = list(enumerate(zip(documents, metadatas)))

    def document_order(item):
        original_index, (_, metadata) = item
        metadata = metadata or {}
        if isinstance(metadata.get("chunk_index"), int):
            return (0, metadata["chunk_index"], original_index)
        page = metadata.get("page")
        return (1, page if isinstance(page, int) else float("inf"), original_index)

    indexed_documents.sort(key=document_order)
    return [
        Document(page_content=content, metadata=metadata or {})
        for _, (content, metadata) in indexed_documents
        if content and content.strip()
    ]


def get_summarization_chain():
    """Create a map-reduce chain suitable for documents larger than one context window."""
    llm = OpenAI(temperature=0, openai_api_key=config.get_openai_api_key())
    map_prompt = PromptTemplate.from_template(
        """Extract the important facts, conclusions, and recommendations from this
document chunk. Be concise and factual so the notes can be combined later.

Chunk:
{text}

Concise notes:"""
    )
    combine_prompt = PromptTemplate.from_template(
        """Combine these notes into a useful document summary. Return exactly these
two labeled sections. The executive summary must be 2-3 sentences. Include 5-10
concise key points when the source material supports that many.

EXECUTIVE SUMMARY:
<summary>

KEY POINTS:
- <key point>

Notes:
{text}"""
    )
    return load_summarize_chain(
        llm,
        chain_type="map_reduce",
        map_prompt=map_prompt,
        combine_prompt=combine_prompt,
    )


def parse_summary_output(output_text):
    """Split the summarization chain's labeled output into structured fields."""
    output_text = output_text.strip()
    match = re.search(
        r"EXECUTIVE SUMMARY:\s*(.*?)\s*KEY POINTS:\s*(.*)",
        output_text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if not match:
        return output_text, []

    executive_summary = " ".join(match.group(1).split())
    key_points = [
        point.strip()
        for point in re.findall(
            r"^\s*(?:[-*•]|\d+[.)])\s+(.+?)\s*$",
            match.group(2),
            flags=re.MULTILINE,
        )
        if point.strip()
    ]
    return executive_summary, key_points[:10]


def summarize_document(collection_name):
    """Return a cached summary, generating and caching one on the first request."""
    cached_summary = get_cached_summary(collection_name)
    if cached_summary:
        return cached_summary

    documents = get_indexed_documents(collection_name)
    if not documents:
        raise ValueError("No indexed document content is available to summarize.")

    result = get_summarization_chain().invoke({"input_documents": documents})
    output_text = result.get("output_text", "")
    executive_summary, key_points = parse_summary_output(output_text)
    if not executive_summary:
        raise ValueError("The summarization model returned an empty summary.")

    cache_summary(collection_name, executive_summary, key_points)
    return get_cached_summary(collection_name)


def build_vector_store(pdf_path, collection_name):
    if not config.is_openai_key_available():
        raise RuntimeError(
            "OpenAI API key is missing. Set OPENAI_API_KEY or OPENAI_ADMIN_KEY and restart the app."
        )

    documents = pdf_processing.extract_text_and_images_from_pdf(pdf_path)
    if not documents:
        raise ValueError("No text could be extracted from the uploaded PDF.")

    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    split_documents = splitter.split_documents(documents)
    for chunk_index, document in enumerate(split_documents):
        document.metadata["chunk_index"] = chunk_index

    embeddings = OpenAIEmbeddings(openai_api_key=config.get_openai_api_key())
    store = Chroma.from_documents(
        split_documents,
        embeddings,
        persist_directory=config.PERSIST_DIR,
        collection_name=collection_name,
    )
    invalidate_cached_summary(collection_name)
    return store


def get_qa_chain(collection_name):
    api_key = config.get_openai_api_key()
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    retriever = Chroma(
        persist_directory=config.PERSIST_DIR,
        embedding_function=embeddings,
        collection_name=collection_name,
    ).as_retriever(search_kwargs={"k": 4})
    llm = OpenAI(temperature=0, openai_api_key=api_key)
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
    )
