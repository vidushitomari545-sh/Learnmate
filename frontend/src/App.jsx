import { useCallback, useEffect, useRef, useState } from "react";
import {
  formatUnexpectedUploadResponse,
  formatUploadFailure,
  getUploadConnectionError,
  isExtractedPdfResponse,
} from "./uploadErrors";

const learningSteps = [
  {
    number: "01",
    title: "Bring your materials together",
    description:
      "Keep your course notes and reading in one place, ready for a focused study session.",
    label: "Planned",
  },
  {
    number: "02",
    title: "Make tricky ideas clearer",
    description:
      "Get help working through concepts and questions as you study.",
    label: "Planned",
  },
  {
    number: "03",
    title: "Practice with purpose",
    description:
      "Use practice and reflection to decide what you would like to revisit.",
    label: "Planned",
  },
];

const maxUploadSizeMegabytes = 10;
const maxUploadSizeBytes = maxUploadSizeMegabytes * 1024 * 1024;

function formatFileSize(sizeBytes) {
  if (sizeBytes < 1024) {
    return `${sizeBytes} B`;
  }

  if (sizeBytes < 1024 * 1024) {
    return `${(sizeBytes / 1024).toFixed(1)} KB`;
  }

  return `${(sizeBytes / (1024 * 1024)).toFixed(1)} MB`;
}

function App() {
  const [connection, setConnection] = useState({
    state: "checking",
    message: "Checking the API connection…",
  });
  const [selectedFile, setSelectedFile] = useState(null);
  const [uploadState, setUploadState] = useState("idle");
  const [uploadMessage, setUploadMessage] = useState("");
  const [documents, setDocuments] = useState([]);
  const [documentsState, setDocumentsState] = useState("loading");
  const [documentsError, setDocumentsError] = useState("");
  const fileInputRef = useRef(null);

  const checkConnection = useCallback(async (signal) => {
    setConnection({ state: "checking", message: "Checking the API connection…" });

    try {
      const response = await fetch("/api/health", { signal });
      if (!response.ok) {
        throw new Error(`The API returned an unexpected response (${response.status}).`);
      }

      const result = await response.json();
      if (result.status !== "ok") {
        throw new Error("The API health check returned an unexpected status.");
      }

      setConnection({ state: "connected", message: "Backend connected" });
    } catch (error) {
      if (error.name === "AbortError") {
        return;
      }

      setConnection({
        state: "disconnected",
        message: "Backend unavailable",
      });
    }
  }, []);

  const loadDocuments = useCallback(async (signal) => {
    setDocumentsState("loading");
    setDocumentsError("");

    try {
      const response = await fetch("/api/documents", { signal });
      if (!response.ok) {
        throw new Error(`Could not load study materials (${response.status}).`);
      }

      let result;
      try {
        result = await response.json();
      } catch {
        throw new Error("The document list response was invalid.");
      }

      if (
        !result ||
        typeof result !== "object" ||
        !Array.isArray(result.documents) ||
        result.documents.some(
          (document) =>
            !document ||
            typeof document.id !== "string" ||
            typeof document.filename !== "string" ||
            typeof document.uploaded_at !== "string" ||
            !Number.isFinite(Date.parse(document.uploaded_at)) ||
            !Number.isSafeInteger(document.size_bytes) ||
            document.size_bytes < 0,
        )
      ) {
        throw new Error("The document list response was invalid.");
      }

      setDocuments(result.documents);
      setDocumentsState("loaded");
    } catch (error) {
      if (error.name === "AbortError") {
        return;
      }

      setDocumentsState("error");
      setDocumentsError(
        error instanceof Error ? error.message : "Could not load study materials.",
      );
    }
  }, []);

  useEffect(() => {
    const controller = new AbortController();
    checkConnection(controller.signal);
    loadDocuments(controller.signal);
    return () => controller.abort();
  }, [checkConnection, loadDocuments]);

  function handleFileChange(event) {
    const file = event.target.files?.[0];
    if (!file) {
      return;
    }

    if (!file.name.toLowerCase().endsWith(".pdf")) {
      event.target.value = "";
      setSelectedFile(null);
      setUploadState("error");
      setUploadMessage("Choose a PDF file to continue.");
      return;
    }

    if (file.size > maxUploadSizeBytes) {
      event.target.value = "";
      setSelectedFile(null);
      setUploadState("error");
      setUploadMessage(
        `PDF files must be ${maxUploadSizeMegabytes} MB or smaller.`,
      );
      return;
    }

    setSelectedFile(file);
    setUploadState("ready");
    setUploadMessage("PDF ready to upload.");
  }

  async function handleUpload() {
    if (!selectedFile || uploadState === "uploading") {
      return;
    }

    setUploadState("uploading");
    setUploadMessage("Uploading your PDF…");

    const formData = new FormData();
    formData.append("file", selectedFile);

    try {
      const response = await fetch("/api/documents/upload", {
        method: "POST",
        body: formData,
      });
      let result = null;
      try {
        result = await response.json();
      } catch {
        if (response.ok) {
          throw new Error(
            formatUnexpectedUploadResponse(response.status, "was not valid JSON"),
          );
        }
      }

      if (!response.ok) {
        throw new Error(
          formatUploadFailure(
            response.status,
            result && typeof result === "object" ? result.detail : undefined,
          ),
        );
      }

      if (!isExtractedPdfResponse(result)) {
        throw new Error(
          formatUnexpectedUploadResponse(
            response.status,
            "did not match the expected validation format",
          ),
        );
      }

      setUploadState("success");
      setUploadMessage(
        result.text_status === "no_text_found"
          ? `${result.filename} was processed, but no selectable text was found.`
          : result.text_status === "partial_text"
            ? `Text was extracted from some pages of ${result.filename}.`
            : `Text extracted from ${result.page_count} pages of ${result.filename}.`,
      );
      setSelectedFile(null);
      if (fileInputRef.current) {
        fileInputRef.current.value = "";
      }
    } catch (error) {
      setUploadState("error");
      setUploadMessage(getUploadConnectionError(error));
    }
  }

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <a className="brand" href="#home" aria-label="LearnMate home">
          <span className="brand-mark" aria-hidden="true">
            <svg viewBox="0 0 32 32" fill="none">
              <path d="M7 8.5c3.5-.8 6.5.1 9 2.5v14c-2.5-2.4-5.5-3.3-9-2.5v-14Z" />
              <path d="M25 8.5c-3.5-.8-6.5.1-9 2.5v14c2.5-2.4 5.5-3.3 9-2.5v-14Z" />
              <path d="M16 11v14" />
            </svg>
          </span>
          <span>learnmate</span>
        </a>

        <div className="sidebar-caption">YOUR SPACE</div>
        <nav className="side-nav" aria-label="Main navigation">
          <a className="nav-link active" href="#home" aria-current="page">
            <svg viewBox="0 0 24 24" aria-hidden="true">
              <path d="m3.5 10 8.5-7 8.5 7v10.5h-6v-7h-5v7h-6V10Z" />
            </svg>
            Overview
          </a>
          <a className="nav-link" href="#study-materials">
            <svg viewBox="0 0 24 24" aria-hidden="true">
              <path d="M5 4.5h14v15H5zM8 8h8M8 12h8M8 16h5" />
            </svg>
            Study materials
          </a>

          <span className="nav-link disabled" aria-disabled="true">
            <svg viewBox="0 0 24 24" aria-hidden="true">
              <path d="M12 3.5 14.2 9l5.8 2-5.8 2L12 19l-2.2-6L4 11l5.8-2L12 3.5Z" />
              <path d="m19 15 .9 2.1L22 18l-2.1.9L19 21l-.9-2.1L16 18l2.1-.9L19 15Z" />
            </svg>
            Study assistant
            <span className="nav-soon">Soon</span>
          </span>
        </nav>

        <div className="sidebar-bottom">
          <div className="sidebar-note">
            <span className="note-icon" aria-hidden="true">✦</span>
            <p>A little progress each day adds up.</p>
          </div>
          <div className="profile">
            <div className="avatar" aria-hidden="true">L</div>
            <div>
              <strong>Your workspace</strong>
              <span>Personal learning</span>
            </div>
          </div>
        </div>
      </aside>

      <main id="home" className="main-content">
        <header className="topbar">
          <div className="breadcrumb">Workspace <span>/</span> Overview</div>
          <div className={`connection-pill ${connection.state}`} role="status">
            <span className="status-dot" aria-hidden="true" />
            <span>{connection.message}</span>
            {connection.state !== "connected" && (
              <button
                className="retry-button"
                type="button"
                onClick={() => checkConnection()}
                disabled={connection.state === "checking"}
              >
                {connection.state === "checking" ? "Checking…" : "Retry"}
              </button>
            )}
          </div>
        </header>

        <div className="page-content">
          <section className="welcome-section">
            <div className="welcome-copy">
              <div className="eyebrow"><span /> A BETTER WAY TO LEARN</div>
              <h1>Your next big idea<br />starts <span>with learning.</span></h1>
              <p className="welcome-description">
                A calm, personal space to explore computer science and make
                sense of new concepts — one step at a time.
              </p>
              <a className="primary-button" href="#getting-started">
                See what’s ahead
                <span aria-hidden="true">↓</span>
              </a>
            </div>
            <div className="welcome-art" aria-hidden="true">
              <div className="art-orbit orbit-one" />
              <div className="art-orbit orbit-two" />
              <div className="art-sun" />
              <div className="art-book">
                <div className="book-page page-left">
                  <span /><span /><span /><span />
                </div>
                <div className="book-page page-right">
                  <span /><span /><span /><span />
                </div>
                <div className="book-spine" />
              </div>
              <div className="art-spark spark-one">✦</div>
              <div className="art-spark spark-two">✧</div>
              <div className="art-dot dot-one" />
              <div className="art-dot dot-two" />
            </div>
            <div className="hero-decoration decoration-top" aria-hidden="true">✳</div>
            <div className="hero-decoration decoration-bottom" aria-hidden="true">✦</div>
          </section>

          <section className="upload-section" aria-labelledby="upload-heading">
            <div className="upload-copy">
              <div className="eyebrow muted">START WITH YOUR MATERIALS</div>
              <h2 id="upload-heading">Upload a study document</h2>
              <p>Choose a PDF to add it to your learning workspace.</p>
            </div>
            <div className="upload-controls">
              <label className="file-picker" htmlFor="pdf-file-input">
                <input
                  id="pdf-file-input"
                  ref={fileInputRef}
                  type="file"
                  accept=".pdf,application/pdf"
                  onChange={handleFileChange}
                  disabled={uploadState === "uploading"}
                />
                <span className="file-picker-icon" aria-hidden="true">↑</span>
                <span className="file-picker-text">
                  <strong>{selectedFile ? selectedFile.name : "Choose a PDF"}</strong>
                  <span>
                    {selectedFile
                      ? "Ready when you are"
                      : `PDF files only · up to ${maxUploadSizeMegabytes} MB`}
                  </span>
                </span>
                <span className="browse-button">Browse files</span>
              </label>
              <button
                className="upload-button"
                type="button"
                onClick={handleUpload}
                disabled={!selectedFile || uploadState === "uploading"}
              >
                {uploadState === "uploading" ? "Uploading…" : "Upload PDF"}
              </button>
            </div>
            <p className={`upload-message ${uploadState}`} role="status" aria-live="polite">
              {uploadMessage || "Your file is checked as a PDF before it is accepted."}
            </p>
          </section>

          <section
            className="documents-section"
            id="study-materials"
            aria-labelledby="documents-heading"
          >
            <div className="section-heading">
              <div>
                <div className="eyebrow muted">YOUR LIBRARY</div>
                <h2 id="documents-heading">Study materials</h2>
              </div>
              {documentsState === "loaded" && documents.length > 0 && (
                <span className="document-count">
                  {documents.length} {documents.length === 1 ? "document" : "documents"}
                </span>
              )}
            </div>

            {documentsState === "loading" && (
              <p className="documents-message" role="status">Loading your documents…</p>
            )}
            {documentsState === "error" && (
              <div className="documents-message error" role="status">
                <span>{documentsError}</span>
                <button
                  className="retry-button"
                  type="button"
                  onClick={() => loadDocuments()}
                >
                  Try again
                </button>
              </div>
            )}
            {documentsState === "loaded" && documents.length === 0 && (
              <p className="documents-message">
                No study materials yet. Upload a PDF to see it here.
              </p>
            )}
            {documentsState === "loaded" && documents.length > 0 && (
              <div className="documents-list">
                {documents.map((document) => (
                  <article className="document-card" key={document.id}>
                    <span className="document-icon" aria-hidden="true">PDF</span>
                    <span className="document-details">
                      <strong title={document.filename}>{document.filename}</strong>
                      <span>
                        {new Date(document.uploaded_at).toLocaleString()}
                        {" · "}
                        {formatFileSize(document.size_bytes)}
                      </span>
                    </span>
                    <span className="document-status">Saved</span>
                  </article>
                ))}
              </div>
            )}
          </section>

          <section className="start-section" id="getting-started">
            <div className="section-heading">
              <div>
                <div className="eyebrow muted">YOUR LEARNING JOURNEY</div>
                <h2>A thoughtful start</h2>
              </div>
              <span className="phase-badge"><span /> STARTING WITH THE BASICS</span>
            </div>

            <div className="journey-grid">
              {learningSteps.map((step) => (
                <article className="journey-card" key={step.number}>
                  <div className="card-topline">
                    <span className="step-number">{step.number}</span>
                    <span className="card-spark" aria-hidden="true">✦</span>
                  </div>
                  <h3>{step.title}</h3>
                  <p>{step.description}</p>
                  <div className="card-footer">
                    <span className="upcoming-dot" />
                    {step.label} · not available yet
                  </div>
                </article>
              ))}
            </div>
          </section>

          <footer className="page-footer">
            <span>Made for curious minds.</span>
            <span>LearnMate <span className="footer-divider">·</span> Your learning, your pace</span>
          </footer>
        </div>
      </main>
    </div>
  );
}

export default App;
