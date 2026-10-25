# LearnMate

LearnMate is a learning platform foundation for computer science students. This first phase provides a React dashboard, a FastAPI health endpoint, and a frontend connection-status check. Document processing, AI features, accounts, and persistent data are intentionally out of scope for now.

## Requirements

- Python 3.10 or newer
- Node.js 18 or newer and npm

## Run locally on Windows

Open two terminals from the project root.

### 1. Start the backend

```powershell
cd backend
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-dev.txt
python -m uvicorn app.main:app --reload
```

The API runs at <http://localhost:8000>. Its health endpoint is <http://localhost:8000/api/health>, and interactive API documentation is at <http://localhost:8000/docs>.

If PowerShell blocks virtual-environment activation, use `.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt` and `.\.venv\Scripts\python.exe -m uvicorn app.main:app --reload` instead.

### 2. Start the frontend

In the second terminal:

```powershell
cd frontend
npm install
npm run dev
```

Open the local URL printed by Vite (usually <http://localhost:5173>). The development server proxies `/api` requests to the backend on port 8000, so start the backend first to see a connected status.

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

Git is not initialized in this workspace yet. After reviewing the files, initialize Git and create the first commit:

```powershell
git init
git add .gitignore README.md docs frontend backend
git status
git commit -m "chore: initialize LearnMate frontend and backend"
```

To publish, create an empty repository named `LearnMate` on GitHub (do not initialize it with a README or license), then connect and push:

```powershell
git branch -M main
git remote add origin https://github.com/<YOUR_USERNAME>/LearnMate.git
git push -u origin main
```

Replace `<YOUR_USERNAME>` with your GitHub username. Review `git status` and the staged diff before committing; do not commit `.env` files or virtual environments.

## Next phase

The natural next step is PDF upload and document processing: accept a file, extract its text, and prepare a clear path toward chunking and retrieval. The current app does not yet process documents or provide AI-generated answers.
