# LearnMate

LearnMate is a learning platform foundation for computer science students. It provides a React dashboard, a FastAPI backend, local PDF upload and text extraction, retrieval over uploaded materials, and an optional local Ollama-backed study chat. Accounts and cloud-backed persistence are not implemented.

## Requirements

- Python 3.10 or newer
- Node.js 18 or newer and npm

### One-command startup on Windows

Install Ollama, then install the model once with `ollama pull llama3.2:3b`. The backend virtual environment (`backend\.venv`) and frontend npm dependencies must also be set up. From the project root, run:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\start-learnmate.ps1
```

The script checks Ollama and the installed model, then starts only the missing backend and frontend services in separate windows. It uses Ollama's HTTP API; the model loads when the first summary or chat request arrives. Press Ctrl+C in each service window to stop it; stop the Ollama window only if the script started it.

## Run locally on Windows

Open two terminals from the project root.

### 1. Start the backend

```powershell
cd backend
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-dev.txt
python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8765
```

The API runs at <http://localhost:8765>. Its health endpoint is <http://localhost:8765/api/health>, and interactive API documentation is at <http://localhost:8765/docs>.

If PowerShell blocks virtual-environment activation, use `.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt` and `.\.venv\Scripts\python.exe -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8765` instead.

### Local Ollama chat

Install Ollama for Windows, keep its local service running when using the manual startup instructions, and download the configured model:

```powershell
ollama pull llama3.2:3b
```

The backend defaults to the `ollama` provider at `http://localhost:11434` with model `llama3.2:3b`. To override these defaults, set the following in `backend/.env`:

```dotenv
LEARNMATE_LLM_PROVIDER=ollama
LEARNMATE_OLLAMA_BASE_URL=http://localhost:11434
LEARNMATE_OLLAMA_MODEL=llama3.2:3b
```

`LEARNMATE_LLM_TIMEOUT_SECONDS` and `LEARNMATE_LLM_MAX_RETRIES` configure Ollama request timeouts and retries. No API key or paid API service is required. Set `LEARNMATE_LLM_PROVIDER=fake` to use the deterministic fake provider instead.

### 2. Start the frontend

In the second terminal:

```powershell
cd frontend
npm install
npm run dev
```

Open the local URL printed by Vite (usually <http://localhost:5173>). The development server proxies `/api` requests to the backend on port 8765, so start the backend first to see a connected status.

## Test the backend

With the backend environment activated:

```powershell
cd backend
python -m pytest
```

## Project structure

```text
LearnMate/
├── backend/       FastAPI application and backend tests
├── docs/          Project documentation
├── frontend/      React + Vite application
├── .gitignore
└── README.md
```

The frontend uses a relative `/api` URL rather than embedding a backend host in the UI. Vite handles local development proxying; this keeps the frontend decoupled from the API's deployment location. The FastAPI app is kept in its own package so future document, quiz, and persistence modules can be added without putting those features in the entry point.

## GitHub workflow

This project is already tracked in a Git repository. After making changes, review the working tree and stage only the files intended for the change:

```powershell
git status
git add <files>
git diff --cached
git commit -m "Describe the change"
```

To publish committed changes, configure the GitHub remote if one is not already set, then push the branch:

```powershell
git remote -v
git remote add origin https://github.com/<YOUR_USERNAME>/LearnMate.git
git push -u origin <branch>
```

Replace `<YOUR_USERNAME>` and `<branch>` with the appropriate values. If the `origin` remote is already configured, do not add it again. Review the staged diff before committing, and do not commit `.env` files or virtual environments.

## PDF upload, storage, text extraction, and cleaning

`POST /api/documents/upload` validates that a file has a `.pdf` filename, an `application/pdf` content type, a `%PDF-` signature, and readable, unencrypted PDF content. Empty files and files larger than 10 MB are rejected. Valid PDFs are stored locally under `backend/data/documents` by default, with generated filenames; document metadata and extracted page text are stored in SQLite.

Selectable text is extracted page by page and cleaned before it is stored. Cleaning normalizes line endings, trims surrounding whitespace, collapses repeated spaces and tabs outside recognized code, limits consecutive blank lines outside fenced code, and preserves punctuation, Unicode, paragraph breaks, and recognized fenced-code whitespace. Pages are numbered starting at 1. Scanned pages without selectable text are retained with empty text; OCR is not performed. The `text_status` is `text_found`, `partial_text`, or `no_text_found` depending on how many pages contain non-whitespace text.

The backend also provides a character-based text chunking service for already-extracted pages. It defaults to 1000-character chunks with 150-character overlap, prefers paragraph and sentence boundaries, and retains chunk IDs, indices, and source page references. Chunking is currently available to backend services; it is not exposed as an API endpoint.

Set `LEARNMATE_DOCUMENT_STORAGE` to select a different local storage directory and `LEARNMATE_DEVELOPMENT_OWNER_ID` to configure the development owner scope. Authentication is not implemented; this configuration is intended for a single-owner local development setup. Existing documents without an owner remain unowned and are excluded from owner-filtered routes until explicitly migrated.

Uploads require an `Idempotency-Key` header. Retrying the same key with the same PDF returns the existing document; reusing it for different PDF content returns HTTP 409. A successful upload returns document metadata and an extraction summary:

```json
{
  "id": "d25b8927-f28d-4c46-9e97-39474ed1d67a",
  "filename": "notes.pdf",
  "uploaded_at": "2026-10-31T10:00:00+00:00",
  "size_bytes": 2048,
  "status": "stored",
  "page_count": 2,
  "text_status": "partial_text"
}
```

`GET /api/health` reports backend health. `GET /api/documents` returns owner-filtered document metadata and a total count, with `limit` and `cursor` pagination. `GET /api/documents/{document_id}/pages` returns the saved 1-based page numbers and cleaned text for an owned document.

## Planned features

OCR, authentication, and cloud-backed storage are not implemented.
