param(
    [Parameter(Mandatory=$true)]
    [string]$SourceDir,
    
    [Parameter(Mandatory=$true)]
    [string]$DestDir
)

# Check if source directory exists
if (-not (Test-Path $SourceDir -PathType Container)) {
    Write-Error "Error: Source directory '$SourceDir' does not exist"
    exit 1
}

# Step 1: Copy folder structure with only default.yml files
Write-Host "Copying config files from $SourceDir to $DestDir..." -ForegroundColor Cyan
New-Item -ItemType Directory -Force -Path $DestDir | Out-Null

Get-ChildItem -Path $SourceDir -Recurse -Filter "default.yml" | ForEach-Object {
    # Get relative path from source directory
    $relPath = $_.FullName.Substring($SourceDir.Length).TrimStart('\')
    $destFile = Join-Path $DestDir $relPath
    
    # Create directory structure
    $destFolder = Split-Path $destFile -Parent
    New-Item -ItemType Directory -Force -Path $destFolder | Out-Null
    
    # Copy config file
    Copy-Item $_.FullName -Destination $destFile
    Write-Host "Copied: $($_.FullName) -> $destFile" -ForegroundColor Green
}

# Step 2: Run training for each config file iteratively
Write-Host ""
Write-Host "Running training jobs sequentially..." -ForegroundColor Cyan

# Activate virtual environment
$venvActivate = Join-Path $PSScriptRoot "..\.nr-env\Scripts\Activate.ps1"
if (Test-Path $venvActivate) {
    & $venvActivate
}

$configFiles = Get-ChildItem -Path $DestDir -Recurse -Filter "default.yml"

foreach ($configFile in $configFiles) {
    $configDir = $configFile.DirectoryName
    
    Write-Host "=========================================" -ForegroundColor Yellow
    Write-Host "Starting training for: $configDir" -ForegroundColor Yellow
    Write-Host "Config file: $($configFile.FullName)" -ForegroundColor Yellow
    Write-Host "=========================================" -ForegroundColor Yellow
    
    # Run training with real-time logging
    $logFile = Join-Path $configDir "output.log"
    
    # Use Start-Process with passthru to capture exit code while streaming output
    $process = Start-Process -FilePath "python" `
        -ArgumentList "training_combined.py", "--config", $configFile.FullName `
        -NoNewWindow `
        -PassThru `
        -RedirectStandardOutput $logFile `
        -RedirectStandardError "$configDir\error.log" `
        -Wait
    
    if ($process.ExitCode -eq 0) {
        Write-Host "Completed successfully: $configDir" -ForegroundColor Green
    } else {
        Write-Host "Failed with exit code $($process.ExitCode): $configDir" -ForegroundColor Red
    }
    Write-Host ""
}

Write-Host "All training jobs completed!" -ForegroundColor Cyan