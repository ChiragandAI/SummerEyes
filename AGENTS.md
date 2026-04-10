# AGENTS.md

## Project overview
SummerEyes is a Streamlit app for summarizing text, PDFs, and audio using a Groq-hosted LLM plus optional RAG context.

## Repository layout
- `FrontEnd/app.py`: main Streamlit UI and interaction flow.
- `BackEnd/pdf_utils.py`: PDF ingestion/chunking/vector index helpers.
- `BackEnd/speechtotext.py`: speech-to-text utility code.
- `SummerEyes_local_db/`: local FAISS index artifacts.
- `screenshots/`: README/demo images.

## Local setup
1. Create a Python environment.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the app:
   ```bash
   streamlit run FrontEnd/app.py
   ```

## Development guidelines
- Keep changes minimal and focused.
- Prefer clear, user-facing error messages over silent failures.
- Avoid committing secrets or API keys.
- If backend logic changes, verify the frontend path that calls it.

## Validation checks
Run these before submitting major changes:
```bash
python -m py_compile FrontEnd/app.py BackEnd/pdf_utils.py BackEnd/speechtotext.py
```

For full app validation, launch Streamlit and test:
- PDF upload and summarization
- Query answering with and without RAG
- Audio transcription and summary generation

## Notes for future agents
- This repo may include generated FAISS artifacts under `SummerEyes_local_db/`; avoid editing them unless intentionally rebuilding the index.
- Keep README screenshots and feature descriptions aligned with UI behavior.
