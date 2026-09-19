from pydantic import BaseModel, Field


class CheckoutRequest(BaseModel):
    plan: str = Field(pattern="^(analyst|team|enterprise)$")


class PortalRequest(BaseModel):
    customer_id: str = Field(min_length=3, max_length=128)
