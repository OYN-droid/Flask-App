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
   ```

   Or export it in your shell:
   ```bash
   export OPENAI_API_KEY="your_api_key"
   ```

4. Run the app:
   ```bash
   python app.py
   ```

5. Open your browser at `http://127.0.0.1:5000`

## Usage

- Upload a PDF handbook.
- Ask questions about the uploaded document.
- The app indexes the PDF into a Chroma vector store and answers queries with OpenAI.
