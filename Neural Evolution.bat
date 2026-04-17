@echo off
setlocal

cd /d "%~dp0"

echo ============================================
echo   Neural Evolution - Flask Application
echo ============================================
echo.

if not exist "app.py" (
    echo ERROR: app.py was not found in %cd%
    pause
    exit /b 1
)

if not exist "requirements.txt" (
    echo ERROR: requirements.txt was not found in %cd%
    pause
    exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
    echo Creating local virtual environment in .venv...
    py -3 -m venv .venv
    if errorlevel 1 (
        python -m venv .venv
    )
)

if not exist ".venv\Scripts\python.exe" (
    echo ERROR: Could not create .venv. Install Python 3.10 or newer and try again.
    pause
    exit /b 1
)

echo Installing/updating dependencies...
".venv\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 (
    echo ERROR: Failed to upgrade pip.
    pause
    exit /b 1
)

".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 (
    echo ERROR: Failed to install dependencies from requirements.txt.
    pause
    exit /b 1
)

set "PORT=%NEURAL_EVOLUTION_PORT%"
if "%PORT%"=="" set "PORT=5000"

echo.
echo Server will be available at: http://127.0.0.1:%PORT%
echo Close this window or press Ctrl+C to stop the server.
echo ============================================
echo.

start "" "http://127.0.0.1:%PORT%"
".venv\Scripts\python.exe" app.py

pause
