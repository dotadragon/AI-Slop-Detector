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
    confidence_band: str | None = None
    confidence_range: list[float] | None = None
    zone_low: float | None = None
    zone_high: float | None = None


class AnalyzeResponse(BaseModel):
    job_id: str
    result: DetectionResultOut | None = None
    status: str = "completed"


class TextCompareRequest(BaseModel):
    specimen_a: str = Field(min_length=10, max_length=50_000)
    specimen_b: str = Field(min_length=10, max_length=50_000)

    @field_validator("specimen_a", "specimen_b")
    @classmethod
    def no_null_bytes(cls, value: str):
        if "\x00" in value:
            raise ValueError("null_bytes_not_allowed")
        return value


class SpecimenMetrics(BaseModel):
    overall_score: float
    confidence_band: str
    burstiness: float
    perplexity: float | None = None
    vocabulary_richness: float
    char_count: int
    word_count: int


class TextComparisonResponse(BaseModel):
    specimen_a: SpecimenMetrics
    specimen_b: SpecimenMetrics
    delta_score: float
    delta_burstiness: float
    delta_perplexity: float | None = None
    ngram_overlap_pct: float
    shared_phrases: list[str]
    verdict: str
