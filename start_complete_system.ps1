#!/usr/bin/env powershell
# Complete TidyMesh Setup Script
# This script starts everything you need

Write-Host "🚀 TidyMesh Complete Setup" -ForegroundColor Green
Write-Host "=========================" -ForegroundColor Green
Write-Host ""

# Get current directory
$currentDir = Get-Location

Write-Host "📁 Working Directory: $currentDir" -ForegroundColor Cyan
Write-Host ""

# Activate virtual environment
Write-Host "🐍 Activating Python environment..." -ForegroundColor Yellow
& ".\.venv\Scripts\Activate.ps1"

Write-Host "✅ Virtual environment activated" -ForegroundColor Green
Write-Host ""

# Start API server in background
Write-Host "🌐 Starting API Server..." -ForegroundColor Yellow
$apiJob = Start-Job -ScriptBlock {
    Set-Location $using:currentDir
    & ".\.venv\Scripts\Activate.ps1"
    python scripts\tidymesh_api_server.py
}

Write-Host "✅ API Server started (Job ID: $($apiJob.Id))" -ForegroundColor Green
Write-Host ""

# Wait for API to be ready
Write-Host "⏳ Waiting for API server to be ready..." -ForegroundColor Yellow
do {
    Start-Sleep -Seconds 2
    try {
        $response = Invoke-RestMethod -Uri "http://localhost:5000" -Method GET -TimeoutSec 5
        $apiReady = $true
    } catch {
        $apiReady = $false
        Write-Host "   Still starting..." -ForegroundColor Gray
    }
} while (-not $apiReady)

Write-Host "✅ API Server is ready!" -ForegroundColor Green
Write-Host ""

# Show status
Write-Host "📊 System Status:" -ForegroundColor Cyan
Write-Host "   API Server: http://localhost:5000" -ForegroundColor White
Write-Host "   Status: Running" -ForegroundColor Green
Write-Host ""

# Show usage options
Write-Host "🎯 How to Use:" -ForegroundColor Cyan
Write-Host ""
Write-Host "📤 Option 1 - Postman:" -ForegroundColor Yellow
Write-Host "   URL: http://localhost:5000/TidyMesh/Sim/v3/run" -ForegroundColor White
Write-Host "   Method: POST" -ForegroundColor White
Write-Host "   Body: {`"n_trucks`": 15, `"n_bins`": 40, `"n_tlights`": 10, `"n_obstacles`": 10, `"steps`": 100000}" -ForegroundColor White
Write-Host ""

Write-Host "🧪 Option 2 - Test Script:" -ForegroundColor Yellow
Write-Host "   python test_postman_params.py" -ForegroundColor White
Write-Host ""

Write-Host "📊 Option 3 - Monitor Status:" -ForegroundColor Yellow
Write-Host "   python monitor_simulation.py" -ForegroundColor White
Write-Host ""

Write-Host "🔍 Option 4 - Standalone Simulation:" -ForegroundColor Yellow
Write-Host "   python TidyMesh_Sim_v3.py" -ForegroundColor White
Write-Host ""

# Interactive menu
Write-Host "🎮 Interactive Menu:" -ForegroundColor Cyan
Write-Host "1. Send test request (like Postman)" -ForegroundColor White
Write-Host "2. Monitor current simulation" -ForegroundColor White
Write-Host "3. Run standalone simulation" -ForegroundColor White
Write-Host "4. Stop API server and exit" -ForegroundColor White
Write-Host ""

do {
    $choice = Read-Host "Select option (1-4)"
    
    switch ($choice) {
        "1" {
            Write-Host "🧪 Sending test request..." -ForegroundColor Yellow
            python test_postman_params.py
        }
        "2" {
            Write-Host "📊 Monitoring simulation..." -ForegroundColor Yellow
            python monitor_simulation.py
        }
        "3" {
            Write-Host "🎯 Running standalone simulation..." -ForegroundColor Yellow
            python TidyMesh_Sim_v3.py
        }
        "4" {
            Write-Host "🛑 Stopping API server..." -ForegroundColor Red
            Stop-Job -Job $apiJob
            Remove-Job -Job $apiJob
            Write-Host "✅ System shutdown complete" -ForegroundColor Green
            exit
        }
        default {
            Write-Host "❌ Invalid option. Please select 1-4." -ForegroundColor Red
        }
    }
    
    Write-Host ""
    Write-Host "Press any key to return to menu..." -ForegroundColor Gray
    $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
    Write-Host ""
    
} while ($true)