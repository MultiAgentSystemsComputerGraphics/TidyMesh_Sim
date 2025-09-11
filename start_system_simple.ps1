# Simple TidyMesh System Launcher
# This script starts the API server and provides basic options

Write-Host "🚀 TidyMesh System Launcher" -ForegroundColor Green
Write-Host "===========================" -ForegroundColor Green
Write-Host ""

# Get current directory
$currentDir = Get-Location
Write-Host "📁 Working Directory: $currentDir" -ForegroundColor Cyan
Write-Host ""

# Activate virtual environment
Write-Host "🐍 Activating Python environment..." -ForegroundColor Yellow
try {
    & ".\.venv\Scripts\Activate.ps1"
    Write-Host "✅ Virtual environment activated" -ForegroundColor Green
} catch {
    Write-Host "❌ Failed to activate virtual environment" -ForegroundColor Red
    Write-Host "Make sure .venv exists and is properly set up" -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host ""

# Start API server in background
Write-Host "🌐 Starting API Server..." -ForegroundColor Yellow
$apiProcess = Start-Process -FilePath "python" -ArgumentList "scripts\tidymesh_api_server.py" -WindowStyle Hidden -PassThru

Write-Host "✅ API Server starting (Process ID: $($apiProcess.Id))" -ForegroundColor Green
Write-Host ""

# Wait for API to be ready
Write-Host "⏳ Waiting for API server to be ready..." -ForegroundColor Yellow
$apiReady = $false
$attempts = 0
$maxAttempts = 15

do {
    Start-Sleep -Seconds 2
    $attempts++
    try {
        $response = Invoke-RestMethod -Uri "http://localhost:5000" -Method GET -TimeoutSec 5
        $apiReady = $true
        Write-Host "✅ API Server is ready!" -ForegroundColor Green
    } catch {
        if ($attempts -lt $maxAttempts) {
            Write-Host "   Still starting... (attempt $attempts/$maxAttempts)" -ForegroundColor Gray
        } else {
            Write-Host "❌ API Server failed to start within timeout" -ForegroundColor Red
            Stop-Process -Id $apiProcess.Id -Force -ErrorAction SilentlyContinue
            Read-Host "Press Enter to exit"
            exit 1
        }
    }
} while (-not $apiReady -and $attempts -lt $maxAttempts)

Write-Host ""

# Show status and options
Write-Host "📊 System Status:" -ForegroundColor Cyan
Write-Host "   API Server: http://localhost:5000" -ForegroundColor White
Write-Host "   Status: Running (PID: $($apiProcess.Id))" -ForegroundColor Green
Write-Host ""

Write-Host "🎯 Available Options:" -ForegroundColor Cyan
Write-Host ""
Write-Host "1. 📤 Send test request (simulates Postman)" -ForegroundColor Yellow
Write-Host "2. 📊 Monitor simulation status" -ForegroundColor Yellow
Write-Host "3. 🎯 Run standalone simulation" -ForegroundColor Yellow
Write-Host "4. 📖 Show Postman instructions" -ForegroundColor Yellow
Write-Host "5. 🛑 Stop and exit" -ForegroundColor Red
Write-Host ""

# Main menu loop
do {
    $choice = Read-Host "Select option (1-5)"
    Write-Host ""
    
    if ($choice -eq "1") {
        Write-Host "🧪 Sending test request..." -ForegroundColor Yellow
        try {
            & python test_postman_params.py
        } catch {
            Write-Host "❌ Error running test script" -ForegroundColor Red
        }
    }
    elseif ($choice -eq "2") {
        Write-Host "📊 Monitoring simulation..." -ForegroundColor Yellow
        try {
            & python monitor_simulation.py
        } catch {
            Write-Host "❌ Error running monitor script" -ForegroundColor Red
        }
    }
    elseif ($choice -eq "3") {
        Write-Host "🎯 Running standalone simulation..." -ForegroundColor Yellow
        try {
            & python TidyMesh_Sim_v3.py
        } catch {
            Write-Host "❌ Error running simulation" -ForegroundColor Red
        }
    }
    elseif ($choice -eq "4") {
        Write-Host "📖 Postman Instructions:" -ForegroundColor Cyan
        Write-Host "========================" -ForegroundColor Cyan
        Write-Host ""
        Write-Host "1. Open Postman" -ForegroundColor White
        Write-Host "2. Create new POST request" -ForegroundColor White
        Write-Host "3. URL: http://localhost:5000/TidyMesh/Sim/v3/run" -ForegroundColor Green
        Write-Host "4. Headers: Content-Type: application/json" -ForegroundColor White
        Write-Host "5. Body (raw JSON):" -ForegroundColor White
        Write-Host '{' -ForegroundColor Gray
        Write-Host '  "n_trucks": 15,' -ForegroundColor Gray
        Write-Host '  "n_bins": 40,' -ForegroundColor Gray
        Write-Host '  "n_tlights": 10,' -ForegroundColor Gray
        Write-Host '  "n_obstacles": 10,' -ForegroundColor Gray
        Write-Host '  "steps": 100000' -ForegroundColor Gray
        Write-Host '}' -ForegroundColor Gray
        Write-Host "6. Click Send" -ForegroundColor White
    }
    elseif ($choice -eq "5") {
        Write-Host "🛑 Stopping API server..." -ForegroundColor Red
        Stop-Process -Id $apiProcess.Id -Force -ErrorAction SilentlyContinue
        Write-Host "✅ System shutdown complete" -ForegroundColor Green
        break
    }
    else {
        Write-Host "❌ Invalid option. Please select 1-5." -ForegroundColor Red
    }
    
    if ($choice -ne "5") {
        Write-Host ""
        Write-Host "Press any key to return to menu..." -ForegroundColor Gray
        $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
        Write-Host ""
        Write-Host "🎯 Available Options:" -ForegroundColor Cyan
        Write-Host "1. Send test request  2. Monitor status  3. Standalone sim  4. Postman help  5. Exit" -ForegroundColor White
        Write-Host ""
    }
    
} while ($true)

Write-Host ""
Write-Host "👋 Thank you for using TidyMesh!" -ForegroundColor Green