param(
  [string]$BaseUrl = "http://127.0.0.1:8000"
)

$ErrorActionPreference = "Stop"

$health = Invoke-RestMethod "$BaseUrl/health"
if ($health.status -ne "ok") {
  throw "Health check failed"
}

$readiness = Invoke-RestMethod "$BaseUrl/api/v1/ops/launch-readiness"
if (-not $readiness.seo.sitemap) {
  throw "Launch readiness endpoint missing SEO evidence"
}

Write-Output "Smoke passed for $BaseUrl"
