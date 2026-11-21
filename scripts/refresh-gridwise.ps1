# Optional entry point for Windows Task Scheduler. Uses local venv + local .env.
$ErrorActionPreference = 'Stop'
$project = Split-Path -Parent $PSScriptRoot
Set-Location $project
$python = Join-Path $project '.venv\Scripts\python.exe'
if (-not (Test-Path $python)) { throw "GridWise Python environment is missing: $python" }
& $python -m gridwise.refresh
if ($LASTEXITCODE -ne 0) { throw "GridWise refresh failed with exit code $LASTEXITCODE" }
