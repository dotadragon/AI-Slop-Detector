import secrets
import time
from collections import defaultdict, deque
from collections.abc import Awaitable, Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

SAFE_METHODS = {"GET", "HEAD", "OPTIONS"}


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
        nonce = secrets.token_urlsafe(16)
        request.state.csp_nonce = nonce
        response = await call_next(request)
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            f"script-src 'self' 'nonce-{nonce}'; "
            "style-src 'self' 'unsafe-inline' fonts.googleapis.com; "
            "font-src 'self' fonts.gstatic.com; "
            "img-src 'self' data: blob:; "
            "connect-src 'self' wss: ws:; "
            "frame-ancestors 'none'; base-uri 'self'; form-action 'self'"
        )
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "0"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = "camera=(self), microphone=(), geolocation=(), payment=()"
        response.headers["Strict-Transport-Security"] = "max-age=63072000; includeSubDomains; preload"
        response.headers["Cross-Origin-Opener-Policy"] = "same-origin"
        response.headers["Cross-Origin-Resource-Policy"] = "same-origin"
        return response


class MethodGuardMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
        if request.method not in {"GET", "POST", "DELETE", "OPTIONS", "HEAD"}:
            return Response(status_code=405, content='{"detail":"method_not_allowed"}', media_type="application/json")
        return await call_next(request)


class CsrfMiddleware(BaseHTTPMiddleware):
    exempt_prefixes = (
        "/api/v1/auth/login",
        "/api/v1/auth/register",
        "/api/v1/auth/forgot-password",
        "/api/v1/auth/reset-password",
        "/api/v1/auth/refresh",
        "/api/v1/auth/csrf",
        "/api/v1/billing/webhook",
        "/api/v1/video/upload",
        "/api/v1/video/upload-stream",
        "/api/v1/image/upload",
        "/api/v1/text/upload",
        "/api/v1/detect/",
        "/api/v1/feedback",
        "/docs",
        "/openapi.json",
    )

    async def dispatch(self, request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
        if request.method not in SAFE_METHODS and request.url.path.startswith("/api/"):
            if not any(request.url.path.startswith(prefix) for prefix in self.exempt_prefixes):
                cookie_token = request.cookies.get("csrf_token")
                header_token = request.headers.get("X-CSRF-Token")
                if not cookie_token or not header_token or not secrets.compare_digest(cookie_token, header_token):
                    return Response(status_code=403, content='{"detail":"csrf_failed"}', media_type="application/json")
        return await call_next(request)


class SlidingWindowDdosMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, max_per_second: int = 100, block_seconds: int = 3600):
        super().__init__(app)
        self.max_per_second = max_per_second
        self.block_seconds = block_seconds
        self.requests: dict[str, deque[float]] = defaultdict(deque)
        self.blocked_until: dict[str, float] = {}
        self.whitelisted_hosts = {"127.0.0.1", "::1", "localhost", "testclient"}

    async def dispatch(self, request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
        ip = request.client.host if request.client else "unknown"
        if ip in self.whitelisted_hosts:
            return await call_next(request)

        now = time.time()
        if self.blocked_until.get(ip, 0) > now:
            return Response(status_code=403, content='{"detail":"ip_blocked"}', media_type="application/json")

        window = self.requests[ip]
        while window and window[0] < now - 1:
            window.popleft()
        window.append(now)
        if len(window) > self.max_per_second:
            self.blocked_until[ip] = now + self.block_seconds
            return Response(status_code=403, content='{"detail":"ip_blocked"}', media_type="application/json")
        return await call_next(request)
