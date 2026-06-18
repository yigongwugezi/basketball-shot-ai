$ErrorActionPreference = "Stop"

$root = Resolve-Path (Join-Path $PSScriptRoot "..")
$outDir = Join-Path $root "datasets\raw\public\video\basketball-51"
New-Item -ItemType Directory -Force $outDir | Out-Null

if (-not $env:KAGGLE_API_TOKEN) {
    $accessTokenPath = Join-Path $HOME ".kaggle\access_token"
    if (-not (Test-Path $accessTokenPath)) {
        Write-Error "Missing Kaggle token. Set KAGGLE_API_TOKEN or save it to $accessTokenPath"
    }
}

python -m kaggle datasets download sarbagyashakya/basketball-51-dataset -p $outDir --unzip

$videos = Get-ChildItem -Recurse -File $outDir |
    Where-Object { $_.Extension -match '^\.(mp4|avi|mov|mkv)$' }

Write-Host "Done: $outDir"
Write-Host "Video files found: $($videos.Count)"

