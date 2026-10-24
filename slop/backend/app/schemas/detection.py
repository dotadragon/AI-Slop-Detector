from pydantic import BaseModel, Field, field_validator


class TextAnalyzeRequest(BaseModel):
    text: str = Field(min_length=1, max_length=50_000)

    @field_validator("text")
    @classmethod
    def no_null_bytes(cls, value: str):
        if "\x00" in value:
            raise ValueError("null_bytes_not_allowed")
        return value


class SentenceScore(BaseModel):
    text: str
    ai_prob: float
    perplexity: float


class DetectionResultOut(BaseModel):
    overall_score: float
    signals: dict
    sentences: list[SentenceScore] | None = None
    frame_scores: list[dict] | None = None
    heatmap_b64: str | None = None
    model_hint: str | None = None


class AnalyzeResponse(BaseModel):
    job_id: str
    result: DetectionResultOut | None = None
    status: str = "completed"
