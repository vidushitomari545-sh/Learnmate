# LearnMate documentation

LearnMate has a React/Vite frontend and a FastAPI backend. The frontend checks `GET /api/health`, and Vite proxies `/api` requests to the local backend.

## Implemented PDF workflow

`POST /api/documents/upload` accepts PDF files and requires an `Idempotency-Key` header. The backend validates the `.pdf` filename, `application/pdf` content type, `%PDF-` signature, readability, encryption state, and a 10 MB size limit. Valid PDFs are stored in the configured local document directory (by default `backend/data/documents`); metadata and extracted page text are stored in SQLite.

Selectable text is extracted page by page and cleaned before persistence. Cleaning normalizes line endings, trims surrounding whitespace, collapses repeated horizontal whitespace outside recognized code, and limits consecutive blank lines outside fenced code while preserving paragraph breaks, symbols, and Unicode. Properly fenced code retains its internal whitespace. Pages remain numbered from 1. Pages without selectable text are retained with empty text; OCR is not performed.

Successful uploads return document metadata, `page_count`, and `text_status` (`text_found`, `partial_text`, or `no_text_found`). The page text is available through:

- `GET /api/documents` — owner-filtered metadata, total count, and `limit`/`cursor` pagination.
- `GET /api/documents/{document_id}/pages` — saved page numbers and cleaned text for an owned document.
- `GET /api/health` — backend health.

The backend has a separate character-based text chunking service for extracted page records. It uses configurable chunk sizes and overlap (defaults: 1000 and 150 characters), prefers paragraph and sentence boundaries, and preserves chunk IDs and source-page references. It is not currently exposed as an API endpoint.

The development owner is configured through `LEARNMATE_DEVELOPMENT_OWNER_ID`. Authentication is not implemented; this is intended for a single-owner local development setup. `LEARNMATE_DOCUMENT_STORAGE` configures the local PDF storage directory. Existing records without an owner remain unowned and hidden from owner-filtered routes until explicitly assigned.

## Local setup

Use Python 3.10 or newer and Node.js 18 or newer with npm. From the project root, start the backend in one terminal:

```powershell
cd backend
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-dev.txt
python -m uvicorn app.main:app --reload
```

In another terminal, start the frontend:

```powershell
cd frontend
npm install
npm run dev
```

The backend runs at <http://localhost:8000>; interactive API documentation is at <http://localhost:8000/docs>. Vite prints the frontend URL, usually <http://localhost:5173>.

Run backend tests from the `backend` directory with `python -m pytest`.

## Planned features

Embeddings, vector search (including FAISS), RAG, OCR, quizzes, authentication, and cloud-backed storage are not implemented.
