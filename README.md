# Flask PDF Q&A App

A simple Flask application for uploading a PDF handbook and asking questions using LangChain and a local Chroma vector store.

## Setup

1. Create and activate a Python virtual environment:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Create a `.env` file in the project root with your OpenAI key:
   ```bash
   echo "OPENAI_API_KEY=your_api_key" > .env
   python -c "import secrets; print('FLASK_SECRET_KEY=' + secrets.token_hex(32))" >> .env
   ```

   Or export it in your shell:
   ```bash
   export OPENAI_API_KEY="your_api_key"
   export FLASK_SECRET_KEY="$(python -c 'import secrets; print(secrets.token_hex(32))')"
   ```

4. Run the app:
   ```bash
   python app.py
   ```

5. Open your browser at `http://127.0.0.1:5000`

The server binds to `127.0.0.1` with debug mode disabled by default. To enable
debug mode explicitly for local development, run:

```bash
FLASK_DEBUG=1 python app.py
```

`FLASK_ENV=development` also enables debug mode. To listen on another interface,
set `FLASK_RUN_HOST`; for example:

```bash
FLASK_RUN_HOST=0.0.0.0 python app.py
```

Do not enable debug mode when listening on a network-accessible interface, because
the interactive debugger can allow arbitrary code execution.

## Running tests

Install the development dependencies and run the pytest suite:

```bash
pip install -r requirements-dev.txt
pytest
```

The tests use temporary storage and mocked OpenAI/Chroma integrations, so they do
not require API access or modify uploaded documents.

## Usage

- Upload a PDF handbook.
- Ask questions about the uploaded document.
- The app indexes the PDF into a Chroma vector store and answers queries with OpenAI.
