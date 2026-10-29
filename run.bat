@echo off
echo ========================================================
echo   Grammar Autocorrector
echo   REVA University - Bindu Singh M (R24EJ024)
echo ========================================================
echo.

IF NOT EXIST ".venv\Scripts\activate.bat" (
    echo [ERROR] Virtual environment not found in .venv.
    echo Please create one and install dependencies:
    echo   python -m venv .venv
    echo   .venv\Scripts\pip install -r requirements.txt
    pause
    exit /b 1
)

echo Activating Python virtual environment (.venv)...
call .venv\Scripts\activate.bat

echo Starting Grammar Autocorrector on http://127.0.0.1:8000 ...
echo Opening default web browser...
start "" "http://127.0.0.1:8000"

echo Running server (Press Ctrl+C to terminate)...
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
