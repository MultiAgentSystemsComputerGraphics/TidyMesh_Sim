@echo off
echo ========================================
echo TidyMesh Simulation - Dual Run Setup
echo ========================================
echo.
echo This will start both the API server and simulation
echo.
echo Instructions:
echo 1. The API server will start in this window
echo 2. A new window will open for the simulation
echo 3. You can then use Postman to send requests to:
echo    POST http://localhost:5000/TidyMesh/Sim/v3/run
echo.
echo Starting API server...
echo.

REM Activate virtual environment and start API server
call .venv\Scripts\activate.bat
cd /d "%~dp0"

REM Start simulation test in a new window
start "TidyMesh Simulation Test" cmd /k "call .venv\Scripts\activate.bat && python test_api_integration.py && pause"

REM Start the API server in this window
echo API Server starting at http://localhost:5000
echo Press Ctrl+C to stop the server
python scripts\tidymesh_api_server.py