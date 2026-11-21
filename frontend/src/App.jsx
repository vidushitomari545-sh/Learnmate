import { useCallback, useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import {
  formatUnexpectedUploadResponse,
  formatUploadFailure,
  getDocumentCountAfterUpload,
  getIdempotencyKey,
  getUploadConnectionError,
  mergeDocumentPage,
  isStoredUploadResponse,
  mergeDocument,
} from "./uploadErrors";
import { requestChatAnswer } from "./chatApi.js";
import {
  formatPageReferences,
  getDocumentSummary,
  startDocumentSummary,
} from "./documentSummaryApi.js";

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
const markdownComponents = {
  table: ({ children }) => (
    <div className="chat-markdown-table">
      <table>{children}</table>
    </div>
  ),
};
const summaryErrorMessages = {
  document_changed: "The PDF changed while this summary was running. Start a new summary.",
  document_unavailable: "This PDF is no longer available. Refresh your study materials and try again.",
  interrupted: "The summary was interrupted. You can safely retry it.",
  malformed_output: "The model returned a summary that could not be validated. Please retry.",
  no_extractable_text: "No selectable text was found in this PDF. Scanned-page OCR is not available yet.",
  provider_authentication: "The configured language model credentials were rejected.",
  provider_failure: "The language model could not generate this summary.",
  provider_rate_limit: "The language model is rate limited. Please try again later.",
  provider_timeout: "The language model timed out while creating the summary.",
  provider_transient: "The language model is temporarily unavailable. Please retry.",
  summary_context_limit: "A complete section exceeded the configured model context limits. No content was silently omitted.",
  summary_failed: "The summary could not be completed. Please try again.",
};

function getSummaryProgressMessage(job) {
  if (job.status === "queued") {
    return "Your full-document summary is queued…";
  }
  if (job.status !== "processing") {
    return "";
  }
  const stages = {
    mapping: "Summarizing document sections",
    reducing: "Combining section summaries",
    finalizing: "Preparing exam-oriented notes",
  };
  const stage = stages[job.phase] || "Preparing your summary";
  return job.progressTotal > 0
    ? `${stage} (${job.progressCurrent} of ${job.progressTotal})…`
    : `${stage}…`;
}

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
  const [activeSection, setActiveSection] = useState(() => {
    const section = window.location.hash.slice(1);
    return ["study-materials", "chat-assistant"].includes(section)
      ? section
      : "home";
  });
  const [connection, setConnection] = useState({
    state: "checking",
    message: "Checking the API connection…",
  });
  const [selectedFile, setSelectedFile] = useState(null);
  const [uploadState, setUploadState] = useState("idle");
  const [uploadMessage, setUploadMessage] = useState("");
  const [documents, setDocuments] = useState([]);
  const [documentCount, setDocumentCount] = useState(0);
  const [nextCursor, setNextCursor] = useState(null);
  const [documentsState, setDocumentsState] = useState("loading");
  const [documentsError, setDocumentsError] = useState("");
  const [chatMessages, setChatMessages] = useState([]);
  const [chatDraft, setChatDraft] = useState("");
  const [chatSubmitting, setChatSubmitting] = useState(false);
  const [chatError, setChatError] = useState("");
  const [summaryDocumentId, setSummaryDocumentId] = useState("");
  const [summaryJob, setSummaryJob] = useState(null);
  const [summarySubmitting, setSummarySubmitting] = useState(false);
  const [summaryError, setSummaryError] = useState("");
  const fileInputRef = useRef(null);
  const idempotencyKeyRef = useRef(null);
  const chatSubmittingRef = useRef(false);
  const chatRequestControllerRef = useRef(null);
  const summarySubmittingRef = useRef(false);
  const summaryRequestControllerRef = useRef(null);

  useEffect(() => {
    function updateActiveSection() {
      const section = window.location.hash.slice(1);
      setActiveSection(
        ["study-materials", "chat-assistant"].includes(section)
          ? section
          : "home",
      );
    }

    window.addEventListener("hashchange", updateActiveSection);
    return () => {
      window.removeEventListener("hashchange", updateActiveSection);
      const controller = chatRequestControllerRef.current;
      chatRequestControllerRef.current = null;
      chatSubmittingRef.current = false;
      controller?.abort();
      const summaryController = summaryRequestControllerRef.current;
      summaryRequestControllerRef.current = null;
      summarySubmittingRef.current = false;
      summaryController?.abort();
    };
  }, []);

  useEffect(() => {
    if (
      !summaryJob ||
      !["queued", "processing"].includes(summaryJob.status)
    ) {
      return undefined;
    }

    const controller = new AbortController();
    async function pollSummary() {
      while (!controller.signal.aborted) {
        await new Promise((resolve) => window.setTimeout(resolve, 1200));
        if (controller.signal.aborted) {
          return;
        }
        try {
          const updatedJob = await getDocumentSummary(
            summaryJob.documentId,
            summaryJob.id,
            { signal: controller.signal },
          );
          if (controller.signal.aborted) {
            return;
          }
          setSummaryJob(updatedJob);
          setSummaryError("");
          if (!["queued", "processing"].includes(updatedJob.status)) {
            return;
          }
        } catch (error) {
          if (error?.name === "AbortError" || controller.signal.aborted) {
            return;
          }
          setSummaryError(
            error instanceof Error
              ? error.message
              : "Could not check the summary status. Retrying…",
          );
        }
      }
    }

    pollSummary();
    return () => controller.abort();
  }, [summaryJob?.documentId, summaryJob?.id, summaryJob?.status]);

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

  const loadDocuments = useCallback(async (signal, cursor = null, append = false) => {
    if (!append) {
      setDocumentsState("loading");
    }
    setDocumentsError("");

    try {
      const query = new URLSearchParams({ limit: "20" });
      if (cursor) {
        query.set("cursor", cursor);
      }
      const response = await fetch(`/api/documents?${query}`, { signal });
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
        !Number.isSafeInteger(result.total_count) ||
        result.total_count < 0 ||
        !(result.next_cursor === null || typeof result.next_cursor === "string") ||
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

      setDocuments((current) =>
        append
          ? mergeDocumentPage(current, result.documents)
          : result.documents,
      );
      setNextCursor(result.next_cursor);
      setDocumentCount(result.total_count);
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
    idempotencyKeyRef.current = null;
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
    idempotencyKeyRef.current = getIdempotencyKey(idempotencyKeyRef.current);

    try {
      const response = await fetch("/api/documents/upload", {
        method: "POST",
        headers: { "Idempotency-Key": idempotencyKeyRef.current },
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

      if (!isStoredUploadResponse(result)) {
        throw new Error(
          formatUnexpectedUploadResponse(
            response.status,
            "did not match the expected validation format",
          ),
        );
      }

      const savedDocument = {
        id: result.id,
        filename: result.filename,
        uploaded_at: result.uploaded_at,
        size_bytes: result.size_bytes,
        page_count: result.page_count,
        text_status: result.text_status,
      };
      setDocuments((current) => mergeDocument(current, savedDocument));
      setDocumentCount((current) =>
        getDocumentCountAfterUpload(current, documents, savedDocument),
      );
      setDocumentsState("loaded");
      setUploadState("success");
      setUploadMessage(
        result.text_status === "no_text_found"
          ? `${result.filename} was saved, but no selectable text was found.`
          : result.text_status === "partial_text"
            ? `${result.filename} was saved; text was extracted from some pages.`
            : `${result.filename} was saved and text was extracted from ${result.page_count} pages.`,
      );
      setSelectedFile(null);
      idempotencyKeyRef.current = null;
      if (fileInputRef.current) {
        fileInputRef.current.value = "";
      }
    } catch (error) {
      setUploadState("error");
      setUploadMessage(getUploadConnectionError(error));
    }
  }

  async function handleDocumentSummary(document) {
    if (summarySubmittingRef.current) {
      return;
    }

    summarySubmittingRef.current = true;
    setSummarySubmitting(true);
    setSummaryDocumentId(document.id);
    setSummaryJob(null);
    setSummaryError("");
    const controller = new AbortController();
    summaryRequestControllerRef.current = controller;

    try {
      const job = await startDocumentSummary(document.id, {
        signal: controller.signal,
      });
      if (!controller.signal.aborted) {
        setSummaryJob(job);
      }
    } catch (error) {
      if (error?.name !== "AbortError") {
        setSummaryError(
          error instanceof Error
            ? error.message
            : "The PDF summary could not be started.",
        );
      }
    } finally {
      if (summaryRequestControllerRef.current === controller) {
        summaryRequestControllerRef.current = null;
        summarySubmittingRef.current = false;
        setSummarySubmitting(false);
      }
    }
  }

  async function handleChatSubmit(event) {
    event.preventDefault();
    const question = chatDraft.trim();
    if (!question || chatSubmittingRef.current) {
      return;
    }

    chatSubmittingRef.current = true;
    setChatSubmitting(true);
    setChatError("");
    setChatDraft("");
    setChatMessages((current) => [
      ...current,
      { id: crypto.randomUUID(), role: "user", content: question },
    ]);

    const controller = new AbortController();
    chatRequestControllerRef.current = controller;
    try {
      const answer = await requestChatAnswer(question, {
        signal: controller.signal,
      });
      setChatMessages((current) => [
        ...current,
        { id: crypto.randomUUID(), role: "assistant", ...answer },
      ]);
    } catch (error) {
      if (error?.name !== "AbortError") {
        setChatError(
          error instanceof Error
            ? error.message
            : "The question could not be sent. Please try again.",
        );
      }
    } finally {
      if (chatRequestControllerRef.current === controller) {
        chatRequestControllerRef.current = null;
        chatSubmittingRef.current = false;
        setChatSubmitting(false);
      }
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
          <a
            className={`nav-link ${activeSection === "home" ? "active" : ""}`}
            href="#home"
            aria-current={activeSection === "home" ? "page" : undefined}
          >
            <svg viewBox="0 0 24 24" aria-hidden="true">
              <path d="m3.5 10 8.5-7 8.5 7v10.5h-6v-7h-5v7h-6V10Z" />
            </svg>
            Overview
          </a>
          <a
            className={`nav-link ${activeSection === "study-materials" ? "active" : ""}`}
            href="#study-materials"
            aria-current={activeSection === "study-materials" ? "page" : undefined}
          >
            <svg viewBox="0 0 24 24" aria-hidden="true">
              <path d="M5 4.5h14v15H5zM8 8h8M8 12h8M8 16h5" />
            </svg>
            Study materials
          </a>

          <a
            className={`nav-link ${activeSection === "chat-assistant" ? "active" : ""}`}
            href="#chat-assistant"
            aria-current={activeSection === "chat-assistant" ? "page" : undefined}
          >
            <svg viewBox="0 0 24 24" aria-hidden="true">
              <path d="M12 3.5 14.2 9l5.8 2-5.8 2L12 19l-2.2-6L4 11l5.8-2L12 3.5Z" />
              <path d="m19 15 .9 2.1L22 18l-2.1.9L19 21l-.9-2.1L16 18l2.1-.9L19 15Z" />
            </svg>
            Chat Assistant
          </a>
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
          <div className="breadcrumb">
            Workspace <span>/</span>
            {activeSection === "study-materials"
              ? "Study materials"
              : activeSection === "chat-assistant"
                ? "Chat Assistant"
                : "Overview"}
          </div>
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
              {documentsState === "loaded" && documentCount > 0 && (
                <span className="document-count">
                  {documentCount} {documentCount === 1 ? "document" : "documents"}
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
            {documentsState === "loaded" && documentCount === 0 && (
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
                    <button
                      className="document-summary-button"
                      type="button"
                      onClick={() => handleDocumentSummary(document)}
                      disabled={
                        summarySubmitting ||
                        (summaryJob &&
                          ["queued", "processing"].includes(summaryJob.status))
                      }
                      aria-label={`Create a full summary of ${document.filename}`}
                    >
                      {summaryDocumentId === document.id &&
                      (summarySubmitting ||
                        (summaryJob &&
                          ["queued", "processing"].includes(summaryJob.status)))
                        ? "Summarizing…"
                        : "Summarize PDF"}
                    </button>
                  </article>
                ))}
              </div>
            )}
            {summaryDocumentId && (
              <section
                className="document-summary-panel"
                aria-labelledby="document-summary-heading"
                aria-live="polite"
              >
                <div className="document-summary-heading">
                  <div>
                    <div className="eyebrow muted">FULL DOCUMENT SUMMARY</div>
                    <h3 id="document-summary-heading">
                      {summaryJob?.filename ||
                        documents.find(
                          (document) => document.id === summaryDocumentId,
                        )?.filename ||
                        "Selected PDF"}
                    </h3>
                  </div>
                  {summaryJob?.cacheHit && (
                    <span className="summary-cache-badge">
                      Saved summary
                    </span>
                  )}
                </div>
                {summarySubmitting && (
                  <p className="summary-status" role="status">
                    Starting your full-document summary…
                  </p>
                )}
                {summaryJob &&
                  ["queued", "processing"].includes(summaryJob.status) && (
                    <div className="summary-progress" role="status">
                      <p>{getSummaryProgressMessage(summaryJob)}</p>
                      <progress
                        max={Math.max(summaryJob.progressTotal, 1)}
                        value={summaryJob.progressCurrent}
                        aria-label="Summary progress"
                      />
                    </div>
                  )}
                {summaryError && (
                  <p className="summary-error" role="alert">
                    {summaryError}
                  </p>
                )}
                {summaryJob?.status === "failed" && (
                  <div className="summary-failure">
                    <p className="summary-error" role="alert">
                      {summaryError ||
                        summaryErrorMessages[summaryJob.errorCode] ||
                        "The summary could not be completed."}
                    </p>
                    {summaryJob.retryable && (
                      <button
                        className="retry-button"
                        type="button"
                        onClick={() =>
                          handleDocumentSummary({
                            id: summaryJob.documentId,
                            filename: summaryJob.filename,
                          })
                        }
                      >
                        Retry summary
                      </button>
                    )}
                  </div>
                )}
                {!summaryJob && !summarySubmitting && summaryError && (
                  <button
                    className="retry-button"
                    type="button"
                    onClick={() =>
                      handleDocumentSummary({
                        id: summaryDocumentId,
                        filename:
                          documents.find(
                            (document) => document.id === summaryDocumentId,
                          )?.filename || "Selected PDF",
                      })
                    }
                  >
                    Try again
                  </button>
                )}
                {summaryJob?.result && (
                  <>
                    {summaryJob.result.textStatus === "partial_text" && (
                      <p className="summary-partial-note">
                        This summary includes all pages with extractable text.
                        No selectable text was found on page
                        {summaryJob.result.pagesWithoutText.length === 1
                          ? ""
                          : "s"}{" "}
                        {formatPageReferences(
                          summaryJob.result.pagesWithoutText,
                        )}
                        .
                      </p>
                    )}
                    <div className="document-summary-content chat-markdown">
                      <ReactMarkdown
                        remarkPlugins={[remarkGfm]}
                        components={markdownComponents}
                      >
                        {summaryJob.result.summaryMarkdown}
                      </ReactMarkdown>
                    </div>
                    <p className="summary-provenance">
                      Source pages:{" "}
                      {formatPageReferences(summaryJob.result.pageReferences)}
                    </p>
                  </>
                )}
              </section>
            )}
            {documentsState === "loaded" && nextCursor && (
              <button
                className="retry-button"
                type="button"
                onClick={() => loadDocuments(undefined, nextCursor, true)}
              >
                Load more documents
              </button>
            )}
          </section>

          <section
            className="chat-section"
            id="chat-assistant"
            aria-labelledby="chat-heading"
          >
            <div className="section-heading">
              <div>
                <div className="eyebrow muted">STUDY WITH YOUR MATERIALS</div>
                <h2 id="chat-heading">Chat Assistant</h2>
              </div>
            </div>
            <div className="chat-panel">
              {chatMessages.length === 0 && !chatSubmitting ? (
                <div className="chat-empty-state">
                  <span className="chat-empty-icon" aria-hidden="true">✦</span>
                  <strong>What would you like to understand?</strong>
                  <p>Ask a question about your uploaded study materials.</p>
                </div>
              ) : (
                <div className="chat-messages" role="log" aria-live="polite">
                  {chatMessages.map((message) => (
                    <article
                      className={`chat-message ${message.role}`}
                      key={message.id}
                    >
                      <strong className="chat-message-author">
                        {message.role === "user" ? "You" : "LearnMate"}
                      </strong>
                      <div
                        className={`chat-message-content ${message.role === "assistant" ? "chat-markdown" : ""}`}
                      >
                        {message.role === "user" ? (
                          message.content
                        ) : (
                          <ReactMarkdown
                            remarkPlugins={[remarkGfm]}
                            components={markdownComponents}
                          >
                            {message.answer}
                          </ReactMarkdown>
                        )}
                      </div>
                      {message.insufficientEvidence && (
                        <p className="chat-evidence-note">
                          I couldn’t find enough evidence in your study materials
                          to answer this confidently.
                        </p>
                      )}
                      {message.citations?.length > 0 && (
                        <div className="chat-citations">
                          <span className="chat-citations-title">Sources</span>
                          {message.citations.map((citation, index) => (
                            <div
                              className="chat-citation"
                              key={`${message.id}-${index}`}
                            >
                              <strong>{citation.documentName}</strong>
                              <span>
                                {citation.pageNumbers.length === 1 ? "Page" : "Pages"}{" "}
                                {citation.pageNumbers.join(", ")}
                              </span>
                            </div>
                          ))}
                        </div>
                      )}
                    </article>
                  ))}
                  {chatSubmitting && (
                    <p className="chat-loading" role="status">
                      <span className="chat-loading-indicator" aria-hidden="true" />
                      LearnMate is thinking…
                    </p>
                  )}
                </div>
              )}
              {chatError && (
                <p className="chat-error" role="alert">{chatError}</p>
              )}
              <form className="chat-composer" onSubmit={handleChatSubmit}>
                <label className="visually-hidden" htmlFor="chat-question">
                  Your question
                </label>
                <textarea
                  id="chat-question"
                  value={chatDraft}
                  onChange={(event) => setChatDraft(event.target.value)}
                  placeholder="Ask a question about your materials…"
                  maxLength={4000}
                  rows={2}
                  disabled={chatSubmitting}
                />
                <button
                  className="chat-send-button"
                  type="submit"
                  disabled={chatSubmitting || !chatDraft.trim()}
                >
                  {chatSubmitting ? "Sending…" : "Send"}
                </button>
              </form>
            </div>
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
