import io
import json
import os
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from langchain_core.documents import Document
from PIL import Image


# app.py requires a session-signing key at import time outside local development.
os.environ.setdefault("FLASK_SECRET_KEY", "pytest-only-secret-key")

import app as app_module
import config
from services import pdf_processing, vector_store


@pytest.fixture
def isolated_app(tmp_path, monkeypatch):
    """Run app storage against temporary directories and a temporary SQLite DB."""
    upload_folder = tmp_path / "uploads"
    audio_folder = tmp_path / "audio"
    persist_folder = tmp_path / "db"
    upload_folder.mkdir()
    audio_folder.mkdir()
    persist_folder.mkdir()

    monkeypatch.setattr(config, "UPLOAD_FOLDER", str(upload_folder))
    monkeypatch.setattr(config, "AUDIO_FOLDER", str(audio_folder))
    monkeypatch.setattr(config, "PERSIST_DIR", str(persist_folder))
    monkeypatch.setattr(
        config, "METADATA_DB_PATH", str(persist_folder / "documents.sqlite3")
    )
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_ADMIN_KEY", raising=False)
    monkeypatch.delenv("SITE_PASSWORD", raising=False)

    vector_store.initialize_metadata_store()
    app_module.app.config.update(
        TESTING=True,
        SECRET_KEY="pytest-session-secret",
        SITE_PASSWORD="",
    )

    return SimpleNamespace(
        app=app_module.app,
        client=app_module.app.test_client(),
        upload_folder=upload_folder,
        audio_folder=audio_folder,
        persist_folder=persist_folder,
    )


def add_document(collection_name="sample", original_filename="Sample.pdf"):
    stored_filename = f"{collection_name}.pdf"
    with vector_store.metadata_connection() as connection:
        connection.execute(
            """
            INSERT INTO documents (collection_name, original_filename, stored_filename)
            VALUES (?, ?, ?)
            """,
            (collection_name, original_filename, stored_filename),
        )
    return stored_filename


@pytest.mark.parametrize(
    ("filename", "expected"),
    [
        ("handbook.pdf", True),
        ("handbook.PDF", True),
        ("archive.pdf.zip", False),
        ("handbook", False),
        ("notes.txt", False),
    ],
)
def test_allowed_file(filename, expected):
    assert config.allowed_file(filename) is expected


@pytest.mark.parametrize(
    ("filename", "expected"),
    [
        ("Report (Final).pdf", "Report__Final_"),
        ("Quarterly-report_2026.pdf", "Quarterly-report_2026"),
        ("résumé.pdf", "r_sum_"),
        ("", "document"),
    ],
)
def test_safe_collection_name(filename, expected):
    assert vector_store.safe_collection_name(filename) == expected


def test_find_uploaded_file_uses_metadata_store(isolated_app):
    stored_filename = add_document("report-123", "Report (Final).pdf")
    expected_path = isolated_app.upload_folder / stored_filename
    expected_path.write_bytes(b"pdf contents")

    assert vector_store.find_uploaded_file("report-123") == str(expected_path)
    assert vector_store.find_uploaded_file("missing") is None


def test_index_returns_200(isolated_app):
    response = isolated_app.client.get("/")

    assert response.status_code == 200


def test_site_password_gate_is_inactive_when_unset(isolated_app, monkeypatch):
    monkeypatch.delenv("SITE_PASSWORD", raising=False)
    ungated_app = app_module.create_app()
    ungated_app.config.update(TESTING=True, SECRET_KEY="pytest-ungated-secret")
    client = ungated_app.test_client()

    assert client.get("/").status_code == 200
    assert client.get("/login").status_code == 302


def test_site_password_gate_requires_successful_login(isolated_app, monkeypatch):
    monkeypatch.setenv("SITE_PASSWORD", "shared-test-password")
    gated_app = app_module.create_app()
    gated_app.config.update(TESTING=True, SECRET_KEY="pytest-gated-secret")
    client = gated_app.test_client()

    protected_response = client.get("/")
    assert protected_response.status_code == 302
    assert protected_response.headers["Location"].endswith("/login")
    assert client.get("/login").status_code == 200
    assert client.get("/static/missing.css").status_code == 404

    rejected_response = client.post(
        "/login", data={"password": "wrong-password"}
    )
    assert rejected_response.status_code == 401
    assert b"Incorrect password." in rejected_response.data

    accepted_response = client.post(
        "/login", data={"password": "shared-test-password"}
    )
    assert accepted_response.status_code == 302
    assert accepted_response.headers["Location"].endswith("/")
    assert client.get("/").status_code == 200


def test_upload_rejects_non_pdf(isolated_app, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    response = isolated_app.client.post(
        "/upload",
        data={"pdf_file": (io.BytesIO(b"not a pdf"), "notes.txt")},
        content_type="multipart/form-data",
        follow_redirects=True,
    )

    assert response.status_code == 200
    assert b"Only PDF files are allowed." in response.data
    assert vector_store.list_documents() == []


def test_upload_rejects_file_without_openai_key(isolated_app):
    response = isolated_app.client.post(
        "/upload",
        data={"pdf_file": (io.BytesIO(b"%PDF-test"), "sample.pdf")},
        content_type="multipart/form-data",
        follow_redirects=True,
    )

    assert response.status_code == 200
    assert b"OpenAI API key is missing." in response.data
    assert vector_store.list_documents() == []


def test_delete_document_removes_collection_and_metadata(isolated_app, monkeypatch):
    stored_filename = add_document("sample", "Sample.pdf")
    uploaded_file = isolated_app.upload_folder / stored_filename
    uploaded_file.write_bytes(b"%PDF-test")
    fake_chroma_client = MagicMock()
    monkeypatch.setattr(vector_store, "chroma_client", fake_chroma_client)

    response = isolated_app.client.post(
        "/delete",
        data={"document_name": "sample"},
        follow_redirects=True,
    )

    assert response.status_code == 200
    fake_chroma_client.delete_collection.assert_called_once_with(name="sample")
    assert vector_store.get_document_metadata("sample") is None
    assert not uploaded_file.exists()
    assert b"Deleted document Sample.pdf." in response.data


def test_ask_renders_answer_with_external_calls_mocked(isolated_app, monkeypatch):
    add_document("sample", "Sample.pdf")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    fake_chain = MagicMock()
    fake_chain.invoke.return_value = {
        "result": "The mocked answer.",
        "source_documents": [
            SimpleNamespace(
                page_content="Page 3:\nThe first supporting passage.",
                metadata={"page": 3, "type": "text"},
            ),
            SimpleNamespace(
                page_content="Page 3:\nA duplicate chunk from the same page.",
                metadata={"page": 3, "type": "text"},
            ),
            SimpleNamespace(
                page_content="Page 7 Image 1:\nA labeled diagram of the workflow.",
                metadata={"page": 7, "type": "image", "image": 1},
            ),
        ],
    }
    get_qa_chain = MagicMock(return_value=fake_chain)
    monkeypatch.setattr(vector_store, "get_qa_chain", get_qa_chain)

    response = isolated_app.client.post(
        "/ask",
        data={"document_name": "sample", "question": "What is this?"},
    )

    assert response.status_code == 200
    get_qa_chain.assert_called_once_with("sample")
    fake_chain.invoke.assert_called_once_with({"query": "What is this?"})
    assert b"What is this?" in response.data
    assert b"The mocked answer." in response.data
    assert b"Sources:" in response.data
    assert response.data.count("Sample.pdf — Page 3</summary>".encode()) == 1
    assert b"Sample.pdf page 3" in response.data
    assert b"Sample.pdf page 7 (image)" in response.data
    assert b"The first supporting passage." in response.data
    assert b"Image description: A labeled diagram of the workflow." in response.data
    assert b"A duplicate chunk from the same page." not in response.data
    assert b"OpenAI API key is missing." not in response.data


def test_ask_multi_merges_ranked_chroma_results_and_cites_documents(
    isolated_app, monkeypatch
):
    add_document("report-q1", "report_q1.pdf")
    add_document("report-q2", "report_q2.pdf")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    q1_collection = MagicMock()
    q1_collection.query.return_value = {
        "documents": [["Q1 page 3 evidence", "Q1 page 2 strongest evidence"]],
        "metadatas": [[{"page": 3, "type": "text"}, {"page": 2, "type": "text"}]],
        "distances": [[0.3, 0.05]],
    }
    q2_collection = MagicMock()
    q2_collection.query.return_value = {
        "documents": [["Q2 page 3 evidence"]],
        "metadatas": [[{"page": 3, "type": "text"}]],
        "distances": [[0.1]],
    }
    fake_chroma_client = MagicMock()
    fake_chroma_client.get_collection.side_effect = lambda name: {
        "report-q1": q1_collection,
        "report-q2": q2_collection,
    }[name]
    monkeypatch.setattr(vector_store, "chroma_client", fake_chroma_client)

    embeddings = MagicMock()
    embeddings.embed_query.return_value = [0.1, 0.2]
    monkeypatch.setattr(
        vector_store, "OpenAIEmbeddings", MagicMock(return_value=embeddings)
    )
    multi_llm = MagicMock()
    multi_llm.invoke.return_value = SimpleNamespace(
        content="The combined mocked answer."
    )
    monkeypatch.setattr(
        vector_store, "get_multi_document_llm", MagicMock(return_value=multi_llm)
    )

    response = isolated_app.client.post(
        "/ask-multi",
        data={
            "document_names": ["report-q1", "report-q2"],
            "question": "Compare the reports.",
        },
    )

    assert response.status_code == 200
    embeddings.embed_query.assert_called_once_with("Compare the reports.")
    fake_chroma_client.get_collection.assert_any_call(name="report-q1")
    fake_chroma_client.get_collection.assert_any_call(name="report-q2")
    q1_collection.query.assert_called_once_with(
        query_embeddings=[[0.1, 0.2]],
        n_results=4,
        include=["documents", "metadatas", "distances"],
    )
    assert b"The combined mocked answer." in response.data
    assert b"report_q1.pdf page 3" in response.data
    assert b"report_q2.pdf page 3" in response.data
    assert b"report_q1.pdf page 2" in response.data
    assert b"Ask across selected" in response.data

    messages = multi_llm.invoke.call_args.args[0]
    context = messages[1].content
    assert context.index("Q1 page 2 strongest evidence") < context.index(
        "Q2 page 3 evidence"
    )
    assert context.index("Q2 page 3 evidence") < context.index("Q1 page 3 evidence")


def test_summarize_document_generates_then_uses_cached_summary(
    isolated_app, monkeypatch
):
    add_document("sample", "Sample.pdf")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    indexed_documents = [
        SimpleNamespace(
            page_content="Page 1:\nIndexed document content.",
            metadata={"page": 1, "type": "text", "chunk_index": 0},
        )
    ]
    get_indexed_documents = MagicMock(return_value=indexed_documents)
    monkeypatch.setattr(
        vector_store, "get_indexed_documents", get_indexed_documents
    )
    fake_llm = MagicMock()
    fake_llm.invoke.side_effect = [
        SimpleNamespace(
            content="This is the first summary sentence. This is the second sentence."
        ),
        SimpleNamespace(
            content=json.dumps(
                [
                    "First key point",
                    "Second key point",
                    "Third key point",
                    "Fourth key point",
                    "Fifth key point",
                ]
            )
        ),
        SimpleNamespace(
            content="This regenerated summary is complete. It replaces the cached result."
        ),
        SimpleNamespace(
            content=json.dumps(
                [
                    "Updated point one",
                    "Updated point two",
                    "Updated point three",
                    "Updated point four",
                    "Updated point five",
                ]
            )
        ),
    ]
    get_summarization_llm = MagicMock(return_value=fake_llm)
    monkeypatch.setattr(
        vector_store, "get_summarization_llm", get_summarization_llm
    )

    first_response = isolated_app.client.get("/summarize/sample")
    second_response = isolated_app.client.get("/summarize/sample")
    regenerated_response = isolated_app.client.post("/summarize/sample/regenerate")

    assert first_response.status_code == 200
    assert second_response.status_code == 200
    assert regenerated_response.status_code == 200
    assert get_summarization_llm.call_count == 2
    assert get_indexed_documents.call_count == 2
    assert fake_llm.invoke.call_count == 4
    assert b"Summary: Sample.pdf" in first_response.data
    assert b"This is the first summary sentence." in first_response.data
    assert b"First key point" in first_response.data
    assert b"Second key point" in second_response.data
    assert b'href="/summarize/sample"' in second_response.data
    assert b'action="/summarize/sample/regenerate"' in second_response.data
    assert b"This regenerated summary is complete." in regenerated_response.data
    assert b"<li>Updated point one</li>" in regenerated_response.data

    cached_summary = vector_store.get_cached_summary("sample")
    assert cached_summary["key_points"] == [
        "Updated point one",
        "Updated point two",
        "Updated point three",
        "Updated point four",
        "Updated point five",
    ]


@pytest.mark.parametrize(
    "model_output",
    [
        "- Not JSON",
        '{"key_points": ["Wrong top-level shape"]}',
        '["Too few points"]',
        '["One", "Two", "Three", "Four", null]',
    ],
)
def test_parse_key_points_json_rejects_malformed_output(model_output):
    with pytest.raises(ValueError):
        vector_store.parse_key_points_json(model_output)


def test_reindex_invalidates_cached_summary(isolated_app, monkeypatch):
    add_document("sample", "Sample.pdf")
    vector_store.cache_summary("sample", "An old summary.", ["Old point"])
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(
        vector_store.pdf_processing,
        "extract_text_and_images_from_pdf",
        MagicMock(
            return_value=[
                Document(
                    page_content="Page 1:\nReplacement content.",
                    metadata={"page": 1, "type": "text"},
                )
            ]
        ),
    )
    monkeypatch.setattr(vector_store, "OpenAIEmbeddings", MagicMock())
    from_documents = MagicMock(return_value=MagicMock())
    monkeypatch.setattr(vector_store.Chroma, "from_documents", from_documents)

    vector_store.build_vector_store("replacement.pdf", "sample")

    assert from_documents.call_count == 1
    assert vector_store.get_cached_summary("sample") is None


@pytest.mark.parametrize(
    ("classification", "structured_content", "expected_subtype", "expected_text"),
    [
        (
            "chart or graph",
            "- Chart type: line chart\n- X-axis: Month\n- Data points: Jan 10, Feb 15\n- Trend: increasing",
            "chart",
            "Data points: Jan 10, Feb 15",
        ),
        (
            "table",
            "| Product | Sales |\n|---|---:|\n| Alpha | 42 |",
            "table",
            "| Alpha | 42 |",
        ),
    ],
)
def test_image_analysis_stores_structured_chart_and_table_content(
    monkeypatch,
    classification,
    structured_content,
    expected_subtype,
    expected_text,
):
    image_buffer = io.BytesIO()
    Image.new("RGB", (20, 20), "white").save(image_buffer, format="PNG")

    stream = MagicMock()
    stream.get_rawdata.return_value = image_buffer.getvalue()
    page = SimpleNamespace(images=[{"stream": stream}])
    page.extract_text = MagicMock(return_value="")
    pdf = SimpleNamespace(pages=[page])
    pdf_context = MagicMock()
    pdf_context.__enter__.return_value = pdf
    monkeypatch.setattr(pdf_processing.pdfplumber, "open", MagicMock(return_value=pdf_context))
    monkeypatch.setattr(
        pdf_processing.pytesseract, "image_to_string", MagicMock(return_value="")
    )

    vision_client = MagicMock()
    vision_client.chat.completions.create.return_value = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content=(
                        '{"classification": '
                        f'"{classification}", "content": '
                        f"{json.dumps(structured_content)}"
                        "}"
                    )
                )
            )
        ]
    )
    monkeypatch.setattr(
        pdf_processing, "OpenAIClient", MagicMock(return_value=vision_client)
    )
    monkeypatch.setattr(config, "get_openai_api_key", MagicMock(return_value="test-key"))

    documents = pdf_processing.extract_text_and_images_from_pdf("sample.pdf")

    assert len(documents) == 1
    assert documents[0].metadata == {
        "page": 1,
        "image": 1,
        "type": "image",
        "subtype": expected_subtype,
    }
    assert expected_text in documents[0].page_content
    assert structured_content in documents[0].page_content
