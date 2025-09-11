@echo off
title TidyMesh Complete System
color 0A

echo.
echo ========================================
echo    TidyMesh Complete System Launcher
echo ========================================
echo.

echo [INFO] Activating Python environment...
call .venv\Scripts\activate.bat

echo [INFO] Starting API Server...
start "TidyMesh API Server" cmd /k "call .venv\Scripts\activate.bat && python scripts\tidymesh_api_server.py"

echo [INFO] Waiting for API server to start...
timeout /t 5 /nobreak > nul

echo.
echo ========================================
echo           System Ready!
echo ========================================
echo.
echo API Server: http://localhost:5000
echo.
echo Choose an option:
echo.
echo 1. Send test request (like Postman)
echo 2. Monitor simulation status  
echo 3. Run standalone simulation
echo 4. Open Postman instructions
echo 5. Exit
echo.

:menu
set /p choice="Enter your choice (1-5): "

if "%choice%"=="1" goto test_request
if "%choice%"=="2" goto monitor
if "%choice%"=="3" goto standalone
if "%choice%"=="4" goto postman_help
if "%choice%"=="5" goto exit
echo Invalid choice. Please try again.
goto menu

:test_request
echo.
echo [INFO] Sending test request...
python test_postman_params.py
echo.
pause
goto menu

:monitor
echo.
echo [INFO] Monitoring simulation...
python monitor_simulation.py
echo.
pause
goto menu

:standalone
echo.
echo [INFO] Running standalone simulation...
python TidyMesh_Sim_v3.py
echo.
pause
goto menu

:postman_help
echo.
echo ========================================
echo          Postman Instructions
echo ========================================
echo.
echo 1. Open Postman
echo 2. Create new POST request
echo 3. URL: http://localhost:5000/TidyMesh/Sim/v3/run
echo 4. Headers: Content-Type: application/json
echo 5. Body (raw JSON):
echo {
echo   "n_trucks": 15,
echo   "n_bins": 40,
echo   "n_tlights": 10,
echo   "n_obstacles": 10,
echo   "steps": 100000
echo }
echo 6. Click Send
echo.
pause
goto menu

:exit
echo.
echo [INFO] Shutting down...
taskkill /f /im python.exe > nul 2>&1
echo [INFO] System shutdown complete.
pause
exit