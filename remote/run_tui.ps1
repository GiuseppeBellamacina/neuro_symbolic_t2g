# ============================================================================
# run_tui.ps1 - Launcher for the t2g TUI (local or remote service).
#
# Usage (from the repo root or anywhere):
#     .\remote\run_tui.ps1
#     .\remote\run_tui.ps1 -Url http://127.0.0.1:8000/t2g
#     .\remote\run_tui.ps1 -Token $myToken
#
# Resolution order:
#   URL:   -Url param > $env:T2G_SERVICE_URL > TUI config screen
#          (NO implicit production default: without a URL the TUI opens
#          its own config screen and saves the values to .env)
#   Token: -Token param > $env:T2G_AUTH_TOKEN > none
#          (optional: the local service runs without auth)
#
# The token is NEVER hardcoded here and is never printed: it comes from
# .env (repo root, gitignored), the process environment, or -Token.
#
# Prerequisite: `uv` installed (https://docs.astral.sh/uv/).
# Stop: Ctrl+C.
# ============================================================================
param(
    [string]$Url,
    [string]$Token
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path $PSScriptRoot -Parent
Push-Location $repoRoot
try {
    # ── Load .env (if present) into the process environment ────────────────
    $envFile = Join-Path $repoRoot ".env"
    if (Test-Path $envFile) {
        Write-Host "[run_tui] Loading .env" -ForegroundColor Cyan
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
        Write-Host "[run_tui] No .env found (ok if T2G_* vars are set elsewhere)" -ForegroundColor DarkGray
    }

    # ── Resolve URL: param > env (nessun default di produzione) ───────────
    if (-not $Url) { $Url = $Env:T2G_SERVICE_URL }

    # ── Resolve token: param > env > nessuno (opzionale in locale) ────────
    if (-not $Token) { $Token = $Env:T2G_AUTH_TOKEN }

    # Senza URL la TUI apre la sua schermata di configurazione (che salva
    # URL+token nel .env): meglio quella di un default implicito che punta
    # alla produzione. Senza token si parte senza auth (servizio locale).
    $tuiArgs = @()
    if ($Url)   { $tuiArgs += "--url";   $tuiArgs += $Url }
    if ($Token) { $tuiArgs += "--token"; $tuiArgs += $Token }

    Write-Host ""
    Write-Host "========================================================" -ForegroundColor Cyan
    Write-Host "  t2g TUI" -ForegroundColor Cyan
    Write-Host "========================================================" -ForegroundColor Cyan
    Write-Host "  URL:   $(if ($Url) { $Url } else { '(da configurare: la TUI apre la schermata di setup)' })"
    Write-Host "  Auth:  $(if ($Token) { 'token loaded (X-Auth-Token header, never printed)' } else { 'no token (ok per il servizio locale senza auth)' })"
    Write-Host "========================================================"
    Write-Host ""

    uv run --extra tui python remote/tui.py @tuiArgs
} finally {
    Pop-Location
}
