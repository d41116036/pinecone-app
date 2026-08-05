from pydantic import BaseModel, Field, field_validator


class PdfUploadResponse(BaseModel):
    filename: str
    page_count: int
    text: str
    char_count: int
    embedding: list[float]
    namespace: str
    chunk_count: int


class RetrieveDocRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=10_000)
    namespace: str = Field(..., min_length=1)
    top_k: int = Field(default=3, ge=1, le=20)

    @field_validator("question", "namespace")
    @classmethod
    def validate_not_blank(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("must not be blank.")
        return cleaned


class RetrieveDocMatch(BaseModel):
    id: str | None = None
    score: float | None = None
    text: str


class RetrieveDocResponse(BaseModel):
    texts: list[str]
    matches: list[RetrieveDocMatch]
    namespace: str


class RetrieveNamespaceResponse(BaseModel):
    namespaces: list[str]
