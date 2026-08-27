# ============================================================================
# serve_local.ps1 — Avvia il servizio di orchestrazione t2g in LOCALE.
#
# Uso (dalla root del repo):
#     .\remote\serve_local.ps1
#     .\remote\serve_local.ps1 -Port 8001
#
# Prerequisiti:
#   - `ssh gcluster 'echo OK'` funziona senza password (alias in ~/.ssh/config;
#     chiave con passphrase → avviare ssh-agent prima).
#   - .env opzionale (copia da .env.example): serve_local lo carica prima di
#     avviare uvicorn, così le env vars arrivano anche al servizio.
#
# Il servizio risponde su http://127.0.0.1:<Port> (default 8000):
#   - GET  /          → health/info (senza auth)
#   - GET  /docs      → documentazione API interattiva (Swagger)
#   - TUI: uv run --extra tui python remote/tui.py --url http://127.0.0.1:8000
#
# Fermare: Ctrl+C.
# ============================================================================
param(
    [int]$Port = 8000
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path $PSScriptRoot -Parent
Push-Location $repoRoot
try {
    # ── Carica .env (se presente) nelle env vars del processo ──────────────
    $envFile = Join-Path $repoRoot ".env"
    if (Test-Path $envFile) {
        Write-Host "[serve_local] Carico .env" -ForegroundColor Cyan
        foreach ($line in Get-Content $envFile) {
            $t = $line.Trim()
            if ($t -and -not $t.StartsWith("#")) {
                $name, $value = $t -split '=', 2
                if ($name -and $null -ne $value) {
                    $name = $name.Trim()
                    $value = $value.Trim().Trim('"')
                    if (-not [Environment]::GetEnvironmentVariable($name)) {
                        Set-Item -Path "Env:$name" -Value $value
                    }
                }
            }
        }
    } else {
        Write-Host "[serve_local] Nessun .env (ok per i default: alias 'gcluster', auth off)" -ForegroundColor DarkGray
    }

    $auth = [bool]$Env:T2G_AUTH_TOKEN
    $host_ = if ($Env:T2G_SSH_HOST) { $Env:T2G_SSH_HOST } else { "gcluster" }

    Write-Host ""
    Write-Host "════════════════════════════════════════════════════════" -ForegroundColor Cyan
    Write-Host "  t2g cluster driver — servizio LOCALE" -ForegroundColor Cyan
    Write-Host "════════════════════════════════════════════════════════" -ForegroundColor Cyan
    Write-Host "  URL:      http://127.0.0.1:$Port"
    Write-Host "  Docs:     http://127.0.0.1:$Port/docs"
    Write-Host "  Cluster:  ssh → $host_ (alias ~/.ssh/config)"
    Write-Host "  Auth:     $(if ($auth) { 'ATTIVA (X-Auth-Token richiesto)' } else { 'disabilitata (solo localhost)' })"
    Write-Host ""
    Write-Host "  TUI:      uv run --extra tui python remote/tui.py --url http://127.0.0.1:$Port"
    Write-Host "  Fermare:  Ctrl+C"
    Write-Host "════════════════════════════════════════════════════════" -ForegroundColor Cyan
    Write-Host ""

    uv run --extra dev uvicorn app:app --host 127.0.0.1 --port $Port --app-dir remote
} finally {
    Pop-Location
}
