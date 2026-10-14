from fastapi import APIRouter, Depends, Header, HTTPException, Request
from starlette.concurrency import run_in_threadpool

from app.api.deps import current_user
from app.core.config import settings
from app.models.user import User
from app.schemas.billing import CheckoutRequest, PortalRequest

try:
    import stripe
except ImportError:
    stripe = None  # type: ignore

router = APIRouter(prefix="/billing", tags=["billing"])


def _price_for_plan(plan: str) -> str:
    return {
        "analyst": settings.stripe_price_analyst,
        "team": settings.stripe_price_team,
        "enterprise": settings.stripe_price_enterprise,
    }.get(plan, "")


@router.post("/checkout")
async def checkout(payload: CheckoutRequest, user: User = Depends(current_user)):
    price_id = _price_for_plan(payload.plan)
    if not settings.stripe_secret_key or not price_id or stripe is None:
        return {
            "mode": "configuration_required",
            "message": "Set STRIPE_SECRET_KEY and install stripe to enable checkout.",
            "plan": payload.plan,
        }

    stripe.api_key = settings.stripe_secret_key
    session = await run_in_threadpool(
        stripe.checkout.Session.create,
        mode="subscription",
        customer_email=user.email,
        line_items=[{"price": price_id, "quantity": 1}],
        success_url=settings.billing_success_url,
        cancel_url=settings.billing_cancel_url,
        metadata={"user_id": str(user.id), "plan": payload.plan},
    )
    return {"url": session.url}


@router.post("/portal")
async def portal(payload: PortalRequest, user: User = Depends(current_user)):
    if not settings.stripe_secret_key or stripe is None:
        return {"mode": "configuration_required", "message": "Set STRIPE_SECRET_KEY to enable the billing portal."}

    stripe.api_key = settings.stripe_secret_key
    session = await run_in_threadpool(
        stripe.billing_portal.Session.create,
        customer=payload.customer_id,
        return_url=str(settings.frontend_url),
    )
    return {"url": session.url, "user_id": str(user.id)}


@router.post("/webhook")
async def stripe_webhook(request: Request, stripe_signature: str | None = Header(default=None, alias="Stripe-Signature")):
    body = await request.body()
    if not settings.stripe_secret_key or not settings.stripe_webhook_secret or stripe is None:
        raise HTTPException(status_code=503, detail="stripe_webhook_not_configured")

    if not stripe_signature:
        raise HTTPException(status_code=400, detail="missing_stripe_signature")

    try:
        event = stripe.Webhook.construct_event(body, stripe_signature, settings.stripe_webhook_secret)
    except Exception as exc:
        raise HTTPException(status_code=400, detail="invalid_webhook") from exc
    return {"received": True, "event_type": event["type"]}
