# ============================================================================
# run_tui.ps1 - Launcher for the t2g TUI (local or remote service).
#
# Usage (from the repo root or anywhere):
#     .\remote\run_tui.ps1
#     .\remote\run_tui.ps1 -Url http://127.0.0.1:8000/t2g
#     .\remote\run_tui.ps1 -Token $myToken
#
# Resolution order:
#   URL:   -Url param > $env:T2G_SERVICE_URL > default (Render manager /t2g)
#   Token: -Token param > $env:T2G_AUTH_TOKEN > error + exit
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
        Write-Host "[run_tui] No .env found (ok if T2G_AUTH_TOKEN is set elsewhere)" -ForegroundColor DarkGray
    }

    # ── Resolve URL: param > env > default ─────────────────────────────────
    if (-not $Url) { $Url = $Env:T2G_SERVICE_URL }
    if (-not $Url) { $Url = "https://render-multi-service-manager.onrender.com/t2g" }

    # ── Resolve token: param > env > error ─────────────────────────────────
    if (-not $Token) { $Token = $Env:T2G_AUTH_TOKEN }
    if (-not $Token) {
        Write-Host "[run_tui] ERROR: no auth token found. Set T2G_AUTH_TOKEN in .env, env var, or pass -Token" -ForegroundColor Red
        exit 1
    }

    Write-Host ""
    Write-Host "========================================================" -ForegroundColor Cyan
    Write-Host "  t2g TUI" -ForegroundColor Cyan
    Write-Host "========================================================" -ForegroundColor Cyan
    Write-Host "  URL:   $Url"
    Write-Host "  Auth:  token loaded (X-Auth-Token header, never printed)"
    Write-Host "========================================================"
    Write-Host ""

    uv run --extra tui python remote/tui.py --url $Url --token $Token
} finally {
    Pop-Location
}
