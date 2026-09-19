from datetime import datetime

from pydantic import BaseModel


class JobStatus(BaseModel):
    job_id: str
    status: str
    progress: int = 0
    created_at: datetime | None = None
    completed_at: datetime | None = None
    result: dict | None = None


class JobHistoryItem(BaseModel):
    job_id: str
    type: str
    score: float | None
    status: str
    created_at: datetime
