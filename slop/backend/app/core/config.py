from functools import lru_cache
from pathlib import Path
from typing import Annotated

from pydantic import AnyHttpUrl, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_env: str = "development"
    app_name: str = "SLOP"
    frontend_url: AnyHttpUrl = "http://localhost:3000"
    allowed_origins: str = "http://localhost:3000"

    database_url: str = "sqlite+aiosqlite:///./slop.db"
    redis_url: str = "redis://localhost:6379/0"

    secret_key: Annotated[str, Field(min_length=16)] = "dev-secret-change-me-dev-secret-change-me"
    jwt_private_key_path: str = "./keys/private.pem"
    jwt_public_key_path: str = "./keys/public.pem"
    access_token_expire_minutes: int = 15
    refresh_token_expire_days: int = 7
    reset_token_expire_minutes: int = 15
    verification_token_expire_hours: int = 24

    smtp_host: str = "localhost"
    smtp_port: int = 1025
    smtp_user: str = ""
    smtp_password: str = ""
    smtp_from: str = "noreply@slop.dev"
    support_email: str = "support@slop.dev"
    marketing_unsubscribe_url: str = "https://slop.dev/unsubscribe"
    sentry_dsn: str = ""
    stripe_secret_key: str = ""
    stripe_webhook_secret: str = ""
    stripe_price_analyst: str = ""
    stripe_price_team: str = ""
    stripe_price_enterprise: str = ""
    billing_success_url: str = "http://localhost:3000/billing?checkout=success"
    billing_cancel_url: str = "http://localhost:3000/billing?checkout=cancelled"

    upload_dir: str = "./uploads"
    max_text_file_mb: int = 20
    max_image_file_mb: int = 50
    max_video_file_mb: int = 500
    huggingface_cache_dir: str = "./models"
    model_root_dir: str = "../Model"
    text_detector_model_id: str = "Hello-SimpleAI/chatgpt-detector-roberta"
    image_detector_model_id: str = "umm-maybe/AI-image-detector"
    video_detector_model_id: str = "dima806/deepfake_vs_real_image_detection"
    image_detector_training_dataset: str = "Real vs AI-generated images (ViT-based classifier)"
    image_live_max_frame_kb: int = 512
    device: str = "auto"

    @property
    def compute_device(self) -> str:
        if self.device == "cpu":
            return "cpu"
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
        except Exception:
            pass
        return "cpu"

    @field_validator("allowed_origins")
    @classmethod
    def no_wildcard_origins(cls, value: str) -> str:
        if "*" in value:
            raise ValueError("ALLOWED_ORIGINS must contain exact origins, never wildcards")
        return value

    @model_validator(mode="after")
    def validate_production_secrets(self) -> "Settings":
        if self.app_env.lower() in {"production", "prod"}:
            if "change-me" in self.secret_key.lower():
                raise ValueError("SECRET_KEY must be explicitly configured and cannot contain 'change-me' in production")
        return self

    @property
    def origins(self) -> list[str]:
        return [origin.strip() for origin in self.allowed_origins.split(",") if origin.strip()]

    @property
    def upload_path(self) -> Path:
        path = Path(self.upload_dir).resolve()
        path.mkdir(parents=True, exist_ok=True)
        return path


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
