param(
  [string]$Container = "slop-postgres-1",
  [string]$Database = "slopdb",
  [string]$User = "slop",
  [string]$OutputDir = ".\backups"
)

$ErrorActionPreference = "Stop"
New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null
$stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$file = Join-Path $OutputDir "slop-$stamp.dump"

docker exec $Container pg_dump -U $User -Fc $Database | Set-Content -Encoding Byte -Path $file
Write-Output "Backup written to $file"
