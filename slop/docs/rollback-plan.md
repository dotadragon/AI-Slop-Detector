# Rollback Plan

## Triggers

- Login, scan, or billing checkout error rate exceeds 2% for 10 minutes.
- Database migration causes failed writes.
- Stripe webhook processing duplicates or drops subscription state.
- Security incident involving uploaded evidence or token handling.

## Steps

1. Freeze deploys.
2. Switch traffic to the previous container image or previous hosting release.
3. Run `scripts/smoke-production.ps1 -BaseUrl https://slop.dev`.
4. If database migration is involved, restore from the latest verified backup or run the documented down migration.
5. Disable billing checkout in env by clearing Stripe price IDs if payment errors persist.
6. Publish incident note on `/status`.
7. Record root cause and prevention action before re-deploying.
