# SLOP

SLOP is a full-stack AI-content signal detector for text, images, and video. It ships with a React + Vite frontend, a FastAPI backend, PostgreSQL models, Redis/Celery scaffolding, Docker Compose, nginx rate-limit configuration, auth flows, and CPU-safe detector implementations.

The bundled detectors are designed to run locally without downloading large model weights. Production hooks are isolated in `backend/app/ml/*` so RoBERTa, timm, Captum, ffmpeg, and deepfake models can be plugged in behind the same response contracts.

## Architecture

```text
Browser
  |
  | React + Vite, Zustand auth, React Query, CSRF header
  v
nginx
  |
  | rate limits, body limits, bad-agent blocking
  v
FastAPI
  |-- auth: RS256 JWT, refresh cookies, reset tokens, 2FA hooks
  |-- security: headers, CORS, CSRF, DDoS sliding window
  |-- detection: text/image/video APIs
  |
  +--> PostgreSQL: users, jobs, results, audit logs
  +--> Redis: intended queue/session/rate-limit backing
  +--> Celery worker: image/video task scaffold
```

## Quick Start

```bash
cd slop
copy .env.example .env
docker compose up --build
```

Frontend: `http://localhost:3000`

Backend: `http://localhost:8000/health`

Launch readiness evidence lives in `docs/launch-readiness.md`. Ruflo coordination is captured in `.claude/workflows/launch-readiness.js` and `.agents/config.toml`; Ruflo coordinates/tracks while Codex performs the implementation.

For a lightweight local backend without Docker:

```bash
cd slop/backend
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

For the frontend:

```bash
cd slop/frontend
npm install
npm run dev
```

## Environment

Important variables live in `.env.example`:

- `DATABASE_URL`: async SQLAlchemy connection string.
- `REDIS_URL`: Redis broker/session URL.
- `SECRET_KEY`: random 64-byte hex secret for app-level signing material.
- `JWT_PRIVATE_KEY_PATH`, `JWT_PUBLIC_KEY_PATH`: RSA key paths. Keys are generated on startup if missing.
- `FRONTEND_URL`, `ALLOWED_ORIGINS`: exact allowed browser origins.
- `UPLOAD_DIR`: private upload storage outside any web root.
- `SMTP_*`: password reset and verification email transport settings.

## API Examples

Get a CSRF token:

```bash
curl -c cookies.txt http://localhost:8000/api/v1/auth/csrf
```

Register:

```bash
curl -b cookies.txt -c cookies.txt -X POST http://localhost:8000/api/v1/auth/register \
  -H "Content-Type: application/json" \
  -d '{"email":"analyst@example.com","password":"ValidPass1!","confirm_password":"ValidPass1!"}'
```

Login:

```bash
curl -b cookies.txt -c cookies.txt -X POST http://localhost:8000/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email":"analyst@example.com","password":"ValidPass1!"}'
```

Analyze text:

```bash
CSRF=$(grep csrf_token cookies.txt | awk '{print $7}')
curl -b cookies.txt -X POST http://localhost:8000/api/v1/text/analyze \
  -H "Content-Type: application/json" \
  -H "X-CSRF-Token: $CSRF" \
  -d '{"text":"Furthermore, it is important to note that this passage is structured and uniform."}'
```

Analyze image:

```bash
curl -b cookies.txt -X POST http://localhost:8000/api/v1/image/analyze \
  -H "X-CSRF-Token: $CSRF" \
  -F "file=@sample.png"
```

Upload video, authenticated:

```bash
curl -b cookies.txt -X POST http://localhost:8000/api/v1/video/upload \
  -H "Authorization: Bearer <access_token>" \
  -H "X-CSRF-Token: $CSRF" \
  -F "file=@sample.mp4"
```

## Security Features

- RS256 JWT access tokens and httpOnly refresh cookies.
- Refresh-token rotation scaffold with family revocation.
- Bcrypt password hashing with cost 12.
- Password reset and email verification token tables.
- TOTP setup and backup-code generation.
- Strict CORS, CSP, HSTS, frame denial, and content-type headers.
- Double-submit CSRF cookie/header validation.
- Honeypot fields and lightweight human-signal checks in frontend forms.
- Upload size limits, UUID storage names, filename normalization, MIME verification with libmagic when available.
- Structured security audit table and auth event logging.
- nginx request/connection limits, method blocking, bad scanner user-agent blocking, and larger video upload route.

## Tests

```bash
cd slop/backend
pytest
```

The tests cover auth smoke flows, honeypot handling, text detection responses, CSRF enforcement, method blocking, and security headers.

## Notes

This repository intentionally keeps heavyweight ML inference behind stable interfaces. `backend/requirements.txt` lists the production libraries requested by the prompt, but the default detector code uses deterministic CPU-safe analysis so the app can run on a normal development machine.
