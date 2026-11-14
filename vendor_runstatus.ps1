# Copies the runstatus library into src/_vendor/runstatus (verbatim, LF line endings).
# The cluster has no access to GitHub: the library travels with the code (sync_cluster / the ClusterControl sync),
# never through pip. Run it after changing runstatus, then commit the result.
#   .\vendor_runstatus.ps1 [-Source ..\runstatus]
param([string]$Source = (Join-Path $PSScriptRoot "..\runstatus"))

$ErrorActionPreference = "Stop"
$src = Join-Path (Resolve-Path $Source) "src\runstatus"
$dst = Join-Path $PSScriptRoot "src\_vendor\runstatus"
if (-not (Test-Path (Join-Path $src "core.py"))) { throw "not a runstatus checkout: $Source" }

New-Item -ItemType Directory -Force $dst | Out-Null
Get-ChildItem $dst -File | Remove-Item
foreach ($f in Get-ChildItem $src -Filter *.py -File) {
    $text = [IO.File]::ReadAllText($f.FullName) -replace "`r`n", "`n"
    [IO.File]::WriteAllText((Join-Path $dst $f.Name), $text, (New-Object Text.UTF8Encoding $false))
}
$version = (Select-String -Path (Join-Path $Source "pyproject.toml") -Pattern '^version\s*=\s*"(.+)"').Matches[0].Groups[1].Value
$commit = (git -C $Source rev-parse --short HEAD)
[IO.File]::WriteAllText((Join-Path $dst "VERSION"), "runstatus $version ($commit)`n")
$pkg = Join-Path $PSScriptRoot "src\_vendor\__init__.py"
if (-not (Test-Path $pkg)) { [IO.File]::WriteAllText($pkg, "# Verbatim copies of external libraries; see vendor_runstatus.ps1.`n") }
Write-Host "vendored runstatus $version ($commit) -> src\_vendor\runstatus"
