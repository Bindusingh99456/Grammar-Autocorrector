#!/usr/bin/env bash
# ========================================================
#   Grammar Autocorrector
#   REVA University - Bindu Singh M (R24EJ024)
# ========================================================

echo "Starting Grammar Autocorrector..."

# Activate virtual environment (support both Linux/macOS and Git Bash)
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
elif [ -f ".venv/Scripts/activate" ]; then
    source .venv/Scripts/activate
else
    echo "[ERROR] Virtual environment not found in .venv."
    echo "Please create one and install dependencies:"
    echo "  python -m venv .venv"
    echo "  pip install -r requirements.txt"
    exit 1
fi

echo "Opening http://127.0.0.1:8000 in your browser..."
if which xdg-open > /dev/null 2>&1; then
    (sleep 2 && xdg-open "http://127.0.0.1:8000") &
elif which open > /dev/null 2>&1; then
    (sleep 2 && open "http://127.0.0.1:8000") &
elif which start > /dev/null 2>&1; then
    (sleep 2 && start "http://127.0.0.1:8000") &
fi

echo "Starting Uvicorn server on http://127.0.0.1:8000 (Press Ctrl+C to stop)..."
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
