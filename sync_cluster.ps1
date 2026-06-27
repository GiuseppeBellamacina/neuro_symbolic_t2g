param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("upload", "download", "download-logs", "download-checkpoints", "download-wandb", "sync-wandb", "push", "pull")]
    [string]$Action,

    [Parameter(Mandatory = $false)]
    [string]$Path  # relative path for push/pull (e.g. "src/training/grpo_t2g_train.py" or "src/cluster/")
)

# ── Cluster connection ────────────────────────────────────────────────────────
$CLUSTER_USER = "bllgpp02h24c351g"
$CLUSTER_HOST = "gcluster.dmi.unict.it"
$REMOTE  = "${CLUSTER_USER}@${CLUSTER_HOST}:~/neuro_symbolic_t2g"
$SSH_TARGET = "${CLUSTER_USER}@${CLUSTER_HOST}"
$LOCAL   = $PSScriptRoot

# Dirs to exclude from download (symlinks on cluster that scp copies as full duplicates)
$TAR_EXCLUDES = @("--exclude=latest", "--exclude=latest-run")

# ── Helpers ───────────────────────────────────────────────────────────────────

function Download-RemoteDir($remoteSubpath, $localDest) {
    New-Item -ItemType Directory -Force -Path $localDest | Out-Null
    $excludeArgs = $TAR_EXCLUDES -join " "
    ssh $SSH_TARGET "cd ~/neuro_symbolic_t2g && tar cf - $excludeArgs $remoteSubpath" | tar xvf - -C "$LOCAL"
}

function Upload {
    Write-Host "Uploading neuro_symbolic_t2g to cluster..." -ForegroundColor Cyan

    # Clean __pycache__ before upload
    Write-Progress -Activity "Upload" -Status "Cleaning __pycache__..." -PercentComplete 0
    Get-ChildItem -Path $LOCAL -Directory -Recurse -Filter "__pycache__" |
        Remove-Item -Recurse -Force -ErrorAction SilentlyContinue

    # Ensure remote directory structure exists
    Write-Progress -Activity "Upload" -Status "Creating remote directories..." -PercentComplete 2
    ssh $SSH_TARGET "mkdir -p ~/neuro_symbolic_t2g/{src,config,logs,checkpoints,data}"

    # Collect all individual files to upload (flatten directories)
    # NOTE: "data" is excluded — ASLG-PC12 is downloaded on the cluster by setup.sh/train.sh
    # NOTE: "logs" and "checkpoints" are excluded — generated on cluster
    $items = @(
        "src",
        "config",
        "grammarllm",
        "main.py",
        "pyproject.toml",
        "README.md",
        ".env"
    )

    # Build flat list of (localPath, remotePath) pairs
    $files = [System.Collections.Generic.List[object]]::new()
    $dirsToClean = [System.Collections.Generic.List[string]]::new()

    foreach ($item in $items) {
        $localPath = Join-Path $LOCAL $item
        if (-not (Test-Path $localPath)) {
            Write-Host "  [SKIP] $item (not found)" -ForegroundColor Yellow
            continue
        }
        if (Test-Path $localPath -PathType Container) {
            # Track top-level dirs for remote cleanup
            $parent = Split-Path $item
            if (-not $parent) {
                $dirsToClean.Add($item)
            }
            # Enumerate all files in directory recursively
            Get-ChildItem -Path $localPath -File -Recurse | ForEach-Object {
                $relPath = $_.FullName.Substring($LOCAL.Length + 1) -replace '\\', '/'
                $files.Add(@{ Local = $_.FullName; Remote = $relPath })
            }
        } else {
            $files.Add(@{ Local = $localPath; Remote = $item })
        }
    }

    # Clean remote top-level dirs before uploading (full replace)
    if ($dirsToClean.Count -gt 0) {
        $rmCmd = ($dirsToClean | ForEach-Object { "rm -rf ~/neuro_symbolic_t2g/$_" }) -join "; "
        ssh $SSH_TARGET $rmCmd
        # Recreate the directories
        $mkCmd = ($dirsToClean | ForEach-Object { "mkdir -p ~/neuro_symbolic_t2g/$_" }) -join "; "
        ssh $SSH_TARGET $mkCmd
    }

    # Ensure all remote subdirectories exist (batch)
    $remoteDirs = $files | ForEach-Object {
        $d = (Split-Path $_.Remote) -replace '\\', '/'
        if ($d) { "~/neuro_symbolic_t2g/$d" }
    } | Sort-Object -Unique
    if ($remoteDirs.Count -gt 0) {
        $mkdirCmd = "mkdir -p " + ($remoteDirs -join " ")
        ssh $SSH_TARGET $mkdirCmd
    }

    # Upload files one by one with granular progress
    $total = $files.Count
    for ($i = 0; $i -lt $total; $i++) {
        $f = $files[$i]
        $pct = [int](($i / $total) * 100)
        $name = $f.Remote
        Write-Progress -Activity "Upload" `
            -Status "[$($i + 1)/$total] $name" `
            -PercentComplete $pct

        scp -q $f.Local "${REMOTE}/$($f.Remote)"
    }

    Write-Progress -Activity "Upload" -Completed
    Write-Host "Upload complete ($total files)." -ForegroundColor Green
}

function DownloadAll {
    Write-Host "Downloading all outputs from cluster..." -ForegroundColor Cyan

    Write-Progress -Activity "Download" -Status "[1/2] logs..." -PercentComplete 0
    DownloadLogs

    Write-Progress -Activity "Download" -Status "[2/2] checkpoints..." -PercentComplete 50
    DownloadCheckpoints

    Write-Progress -Activity "Download" -Completed
    Write-Host "Download complete." -ForegroundColor Green
}

function DownloadLogs {
    Write-Progress -Activity "Download" -Status "Downloading logs/..." -PercentComplete 0
    $dest = Join-Path $LOCAL "logs"
    Download-RemoteDir "logs" $dest
    Write-Progress -Activity "Download" -Completed
    Write-Host "  -> saved to logs\" -ForegroundColor Gray
}

function DownloadCheckpoints {
    Write-Progress -Activity "Download" -Status "Downloading checkpoints/..." -PercentComplete 0
    $dest = Join-Path $LOCAL "checkpoints"
    Download-RemoteDir "checkpoints" $dest
    Write-Progress -Activity "Download" -Completed
    Write-Host "  -> saved to checkpoints\" -ForegroundColor Gray
}

function DownloadWandb {
    # wandb offline runs are saved inside logs/ (e.g. logs/wandb/)
    Write-Progress -Activity "Download" -Status "Downloading wandb offline runs..." -PercentComplete 0

    $dest = Join-Path $LOCAL "logs"
    New-Item -ItemType Directory -Force -Path $dest | Out-Null
    Download-RemoteDir "logs" $dest

    Write-Progress -Activity "Download" -Completed
    Write-Host "  -> saved wandb runs to logs\" -ForegroundColor Gray
    Write-Host ""
    Write-Host "To sync offline runs to wandb.ai:" -ForegroundColor Yellow
    Write-Host "  .\sync_cluster.ps1 -Action sync-wandb" -ForegroundColor Yellow
}

function SyncWandb {
    Write-Host "Syncing wandb offline runs to wandb.ai..." -ForegroundColor Cyan

    # Ensure wandb CLI is available — activate venv if needed
    $venvActivate = Join-Path $LOCAL ".venv\Scripts\Activate.ps1"
    if (-not (Get-Command wandb -ErrorAction SilentlyContinue)) {
        if (Test-Path $venvActivate) {
            Write-Host "  Activating .venv for wandb CLI..." -ForegroundColor Gray
            & $venvActivate
        }
        if (-not (Get-Command wandb -ErrorAction SilentlyContinue)) {
            Write-Host "wandb CLI not found. Install it with: pip install wandb" -ForegroundColor Red
            return
        }
    }

    $logsDir = Join-Path $LOCAL "logs"
    if (-not (Test-Path $logsDir)) {
        Write-Host "No logs/ found. Run download-wandb first." -ForegroundColor Red
        return
    }

    # Find all wandb/ directories with offline run subdirs
    $wandbDirs = Get-ChildItem -Path $logsDir -Recurse -Directory -Filter "wandb" |
        Where-Object { (Get-ChildItem -Path $_.FullName -Directory -Filter "offline-run-*").Count -gt 0 }

    if ($wandbDirs.Count -eq 0) {
        Write-Host "No offline runs found in logs\" -ForegroundColor Yellow
        return
    }

    # Count total offline runs
    $totalRuns = 0
    foreach ($wdir in $wandbDirs) {
        $totalRuns += (Get-ChildItem -Path $wdir.FullName -Directory -Filter "offline-run-*").Count
    }
    Write-Host "Found $totalRuns offline run(s) in $($wandbDirs.Count) wandb dir(s):" -ForegroundColor Gray
    $synced = 0
    $failed = 0
    foreach ($wdir in $wandbDirs) {
        $offlineRuns = Get-ChildItem -Path $wdir.FullName -Directory -Filter "offline-run-*"
        foreach ($run in $offlineRuns) {
            Write-Host "  [$($synced + $failed + 1)/$totalRuns] Syncing $($run.Name) ..." -ForegroundColor Gray -NoNewline
            $result = & wandb sync --include-synced $run.FullName 2>&1
            if ($LASTEXITCODE -eq 0) {
                Write-Host " OK" -ForegroundColor Green
                $synced++
            } else {
                Write-Host " FAILED" -ForegroundColor Red
                Write-Host ($result | Out-String) -ForegroundColor DarkRed
                $failed++
            }
        }
    }

    Write-Host ""
    Write-Host "Sync complete: $synced succeeded, $failed failed." -ForegroundColor $(if ($failed -gt 0) { "Yellow" } else { "Green" })
}

function Push {
    if (-not $Path) {
        Write-Host "Usage: .\sync_cluster.ps1 -Action push -Path <file-or-folder>" -ForegroundColor Red
        return
    }
    $localPath = Join-Path $LOCAL $Path
    if (-not (Test-Path $localPath)) {
        Write-Host "Not found: $Path" -ForegroundColor Red
        return
    }
    $remotePath = $Path -replace '\\', '/'
    # Ensure remote parent directory exists
    $remoteDir = ($remotePath | Split-Path) -replace '\\', '/'
    if ($remoteDir) {
        ssh $SSH_TARGET "mkdir -p ~/neuro_symbolic_t2g/$remoteDir"
    }
    if (Test-Path $localPath -PathType Container) {
        ssh $SSH_TARGET "mkdir -p ~/neuro_symbolic_t2g/$remotePath"
        scp -rq "$localPath/." "${REMOTE}/$remotePath/"
    } else {
        scp -q $localPath "${REMOTE}/$remotePath"
    }
    Write-Host "Pushed $Path -> cluster" -ForegroundColor Green
}

function Pull {
    if (-not $Path) {
        Write-Host "Usage: .\sync_cluster.ps1 -Action pull -Path <file-or-folder>" -ForegroundColor Red
        return
    }
    $remotePath = $Path -replace '\\', '/'
    $localPath = Join-Path $LOCAL $Path
    $localDir = Split-Path $localPath
    if ($localDir) {
        New-Item -ItemType Directory -Force -Path $localDir | Out-Null
    }
    scp -rq "${REMOTE}/$remotePath" $localPath
    Write-Host "Pulled $Path <- cluster" -ForegroundColor Green
}

# ── Dispatch ──────────────────────────────────────────────────────────────────
switch ($Action) {
    "upload"                { Upload }
    "download"              { DownloadAll }
    "download-logs"         { DownloadLogs }
    "download-checkpoints"  { DownloadCheckpoints }
    "download-wandb"        { DownloadWandb }
    "sync-wandb"            { SyncWandb }
    "push"                  { Push }
    "pull"                  { Pull }
}
