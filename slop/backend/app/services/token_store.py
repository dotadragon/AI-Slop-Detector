import secrets
import time
from dataclasses import dataclass

from app.core.security import sha256_token


@dataclass
class RefreshRecord:
    user_id: str
    family_id: str
    expires_at: float
    revoked: bool = False


class InMemoryTokenStore:
    def __init__(self) -> None:
        self.refresh_tokens: dict[str, RefreshRecord] = {}
        self.families: dict[str, set[str]] = {}
        self.csrf_tokens: dict[str, str] = {}
        self.progress: dict[str, int] = {}

    def store_refresh(self, raw_token: str, user_id: str, family_id: str, ttl_seconds: int) -> None:
        token_hash = sha256_token(raw_token)
        self.refresh_tokens[token_hash] = RefreshRecord(user_id=user_id, family_id=family_id, expires_at=time.time() + ttl_seconds)
        self.families.setdefault(family_id, set()).add(token_hash)

    def consume_refresh(self, raw_token: str) -> RefreshRecord | None:
        token_hash = sha256_token(raw_token)
        record = self.refresh_tokens.get(token_hash)
        if not record or record.revoked or record.expires_at < time.time():
            return None
        record.revoked = True
        return record

    def revoke_family(self, family_id: str) -> None:
        for token_hash in self.families.get(family_id, set()):
            if token_hash in self.refresh_tokens:
                self.refresh_tokens[token_hash].revoked = True

    def revoke_user(self, user_id: str) -> None:
        for record in self.refresh_tokens.values():
            if record.user_id == user_id:
                record.revoked = True

    def new_session_csrf(self, user_id: str) -> str:
        token = secrets.token_urlsafe(32)
        self.csrf_tokens[user_id] = token
        return token


token_store = InMemoryTokenStore()
