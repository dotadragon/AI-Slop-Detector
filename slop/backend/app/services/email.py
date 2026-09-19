import structlog

logger = structlog.get_logger()


async def send_email(to_email: str, subject: str, body: str) -> None:
    logger.info("email_queued", to=to_email, subject=subject, body_preview=body[:120])
