from pydantic import BaseModel, ConfigDict, Field, field_validator


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=1, max_length=4000)

    @field_validator("question")
    @classmethod
    def validate_question(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("question must not be empty.")
        return normalized


class ChatCitation(BaseModel):
    document_name: str
    page_numbers: list[int]


class ChatResponse(BaseModel):
    answer: str
    insufficient_evidence: bool
    citations: list[ChatCitation]
