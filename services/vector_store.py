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
from werkzeug.utils import secure_filename

import config
from services import pdf_processing

try:
    from langchain_text_splitters import CharacterTextSplitter
except ImportError:  # LangChain < 0.3
    from langchain.text_splitter import CharacterTextSplitter

try:
    from langchain_classic.chains import RetrievalQA
except ImportError:  # LangChain < 1.0
    from langchain.chains import RetrievalQA


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
            "DELETE FROM documents WHERE collection_name = ?",
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
    return RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
