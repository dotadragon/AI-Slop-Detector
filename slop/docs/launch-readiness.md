# SLOP Launch Readiness

Updated: March 2026 checklist pass.

## Must-Have Status

| Category | Must-have | Status | Evidence |
|---|---|---:|---|
| Analytics | Product analytics installed | Implemented | `posthog-js`, consent-gated `VITE_POSTHOG_KEY` |
| Analytics | Activation and retention events defined | Implemented | `signup_completed`, `scan_started`, `scan_completed`, `checkout_started`, `subscription_changed`, `onboarding_completed` |
| Analytics | Google Search Console verified and sitemap submitted | External | Submit `https://slop.dev/sitemap.xml` in GSC |
| Analytics | Funnel tracking signup to activation to conversion | Implemented | signup, scan, checkout/subscription events |
| SEO | XML sitemap generated | Implemented | `frontend/public/sitemap.xml` |
| SEO | robots.txt configured | Implemented | `frontend/public/robots.txt` |
| SEO | OpenGraph images set | Implemented | `/og-image.svg`, route SEO utility |
| SEO | Meta titles and descriptions | Implemented | `useSeo` on core routes |
| Branding | Favicon set | Implemented | SVG, 16x16, 32x32 variants |
| Legal | Privacy Policy published | Implemented | `/privacy` |
| Legal | Terms of Service published | Implemented | `/terms` |
| Legal | Cookie consent banner | Implemented | consent-gated analytics banner |
| Security | HTTPS everywhere | Implemented for prod | `nginx/nginx.prod.conf` redirects HTTP to HTTPS |
| Security | Security headers | Implemented | FastAPI middleware and nginx headers |
| Security | Input validation and sanitization | Implemented | Pydantic, bleach, MIME checks |
| Security | Rate limiting on API/auth | Implemented | nginx and app-layer DDoS guard |
| Security | Secrets out of codebase | Implemented | `.env.example`, no real secrets committed |
| Email | Support email set up | External | Configure `support@slop.dev` routing |
| Email | Transactional provider | External-ready | SMTP env vars; Resend/Postmark/SES can back them |
| Email | SPF/DKIM/DMARC | External | See `docs/email-dns.md` |
| Email | Password reset tested end to end | Partially automated | Backend flow test coverage; SMTP provider still external |
| Email | Unsubscribe link | Implemented | `MARKETING_UNSUBSCRIBE_URL` in welcome email |
| Monitoring | Error tracking live | External-ready | `SENTRY_DSN` support in backend, frontend env documented |
| Monitoring | Uptime monitoring | External | Point BetterStack/UptimeRobot at `/health` |
| Monitoring | Database backup strategy | Implemented | `scripts/backup-postgres.ps1`, restore notes |
| Billing | Payment provider integrated | External-ready | Stripe checkout/portal/webhook endpoints |
| Billing | Upgrade/downgrade flows tested | External-ready | Stripe portal endpoint; requires test Stripe keys |
| Billing | Failed payment and dunning | External-ready | Stripe webhook endpoint; configure Stripe Billing dunning |
| Launch | Onboarding tested with a real person | External | Use `docs/onboarding-test.md` |
| Launch | Production smoke-tested end to end | Implemented | `scripts/smoke-production.ps1` |
| Launch | Rollback plan | Implemented | `docs/rollback-plan.md` |

## Measurement Readiness Index

Score: 82 / 100, verdict: Usable with Gaps.

- Decision alignment: 22/25. Events map to launch decisions.
- Event clarity: 18/20. Event names are meaningful and stable.
- Data accuracy: 14/20. Needs production validation with real PostHog project.
- Conversion quality: 13/15. Checkout/subscription events distinguish intent and conversion.
- Attribution: 7/10. UTMs are not yet normalized server-side.
- Governance: 8/10. Tracking plan exists here; owner still must be assigned.

## Tracking Plan

| Event | Description | Properties | Trigger | Decision Supported |
|---|---|---|---|---|
| `page_view` | Core page viewed | `path`, `title` | Route SEO hook | Traffic and activation context |
| `cta_clicked` | Primary CTA clicked | `section`, `target` | Hero/cookie CTAs | CTA effectiveness |
| `signup_started` | Register form submitted | none | Register submit | Signup intent |
| `signup_completed` | Registration succeeded | `enable2fa` | Register success | Account activation |
| `login_completed` | Login succeeded | none | Login success | Returning-user health |
| `scan_started` | Scan submitted | `mode`, `chars` | Detector submit/upload | Activation |
| `scan_completed` | Scan result returned | `mode`, `score` | Detector success | Core value delivery |
| `checkout_started` | Billing checkout requested | `plan` | Billing page | Conversion intent |
| `subscription_changed` | Stripe subscription webhook processed | `plan`, `status` | Stripe webhook | Revenue conversion |
| `onboarding_completed` | Human onboarding checklist complete | `tester_type` | Manual/test event | Launch readiness |
