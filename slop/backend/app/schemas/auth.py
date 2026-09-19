from pydantic import BaseModel, ConfigDict, EmailStr, Field, field_validator

from app.core.security import password_is_strong


class HoneypotMixin(BaseModel):
    website: str | None = Field(default=None, max_length=256)

    @property
    def is_honeypot(self) -> bool:
        return bool(self.website)


class RegisterRequest(HoneypotMixin):
    email: EmailStr
    password: str = Field(min_length=8, max_length=128)
    confirm_password: str = Field(min_length=8, max_length=128)
    enable_2fa: bool = False

    @field_validator("confirm_password")
    @classmethod
    def passwords_match(cls, value: str, info):
        if value != info.data.get("password"):
            raise ValueError("passwords_do_not_match")
        if not password_is_strong(value):
            raise ValueError("password_must_include_uppercase_number_and_special")
        return value


class LoginRequest(HoneypotMixin):
    email: EmailStr
    password: str
    totp_code: str | None = Field(default=None, min_length=6, max_length=12)


class ForgotPasswordRequest(HoneypotMixin):
    email: EmailStr


class ResetPasswordRequest(HoneypotMixin):
    token: str = Field(min_length=20, max_length=256)
    new_password: str = Field(min_length=8, max_length=128)

    @field_validator("new_password")
    @classmethod
    def strong_password(cls, value: str):
        if not password_is_strong(value):
            raise ValueError("password_must_include_uppercase_number_and_special")
        return value


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class UserOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    email: EmailStr
    is_active: bool
    is_verified: bool
    totp_enabled: bool


class VerifyTokenRequest(BaseModel):
    token: str


class Verify2FARequest(BaseModel):
    code: str = Field(min_length=6, max_length=12)
