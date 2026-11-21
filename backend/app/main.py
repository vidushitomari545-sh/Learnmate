from contextlib import asynccontextmanager

from fastapi import FastAPI


from app.api.chat import router as chat_router
from app.api.documents import router as documents_router
from app.services.dependencies import (
    get_document_service,
    get_document_summary_job_service,
)


@asynccontextmanager
async def lifespan(_: FastAPI):
    get_document_service().recover_pending_uploads()
    summary_jobs = get_document_summary_job_service()
    summary_jobs.recover_pending_jobs()
    try:
        yield
    finally:
        summary_jobs.close()


app = FastAPI(
    title="LearnMate API",
    description="API foundation for the LearnMate learning platform.",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(documents_router)
app.include_router(chat_router)

@app.get("/api/health", tags=["health"])
def health_check() -> dict[str, str]:
    """Report that the API process is ready to receive requests."""
    return {"status": "ok"}
