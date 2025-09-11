# TidyMesh Simulation - PowerShell Dual Run Setup
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "TidyMesh Simulation - Dual Run Setup" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "This will start both the API server and simulation" -ForegroundColor Yellow
Write-Host ""
Write-Host "Instructions:" -ForegroundColor Green
Write-Host "1. The API server will start in this window" -ForegroundColor White
Write-Host "2. A new window will open for the simulation test" -ForegroundColor White
Write-Host "3. You can then use Postman to send requests to:" -ForegroundColor White
Write-Host "   POST http://localhost:5000/TidyMesh/Sim/v3/run" -ForegroundColor Cyan
Write-Host ""
Write-Host "Starting setup..." -ForegroundColor Yellow
Write-Host ""

# Get script directory
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptDir

# Activate virtual environment
Write-Host "Activating virtual environment..." -ForegroundColor Yellow
& ".\.venv\Scripts\Activate.ps1"

# Start simulation test in a new PowerShell window
Write-Host "Opening simulation test window..." -ForegroundColor Yellow
Start-Process powershell -ArgumentList "-NoExit", "-Command", "& '$scriptDir\.venv\Scripts\Activate.ps1'; python '$scriptDir\test_api_integration.py'; Write-Host 'Press any key to close...'; Read-Host"

# Give the test window a moment to start
Start-Sleep -Seconds 2

# Start the API server in this window
Write-Host "Starting API server at http://localhost:5000" -ForegroundColor Green
Write-Host "Press Ctrl+C to stop the server" -ForegroundColor Red
Write-Host ""

try {
    python scripts\tidymesh_api_server.py
} catch {
    Write-Host "Error starting API server: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host "Make sure Python and dependencies are installed" -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
}