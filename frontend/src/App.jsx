import { useCallback, useEffect, useState } from "react";

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

function App() {
  const [connection, setConnection] = useState({
    state: "checking",
    message: "Checking the API connection…",
  });

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

  useEffect(() => {
    const controller = new AbortController();
    checkConnection(controller.signal);
    return () => controller.abort();
  }, [checkConnection]);

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
          <span className="nav-link disabled" aria-disabled="true">
            <svg viewBox="0 0 24 24" aria-hidden="true">
              <path d="M5 4.5h14v15H5zM8 8h8M8 12h8M8 16h5" />
            </svg>
            Study materials
            <span className="nav-soon">Soon</span>
          </span>
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
