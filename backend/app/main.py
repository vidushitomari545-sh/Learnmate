from fastapi import FastAPI

app = FastAPI(
    title="LearnMate API",
    description="API foundation for the LearnMate learning platform.",
    version="0.1.0",
)


@app.get("/api/health", tags=["health"])
def health_check() -> dict[str, str]:
    """Report that the API process is ready to receive requests."""
    return {"status": "ok"}
