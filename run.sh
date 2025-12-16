#!/bin/bash
# Quick start script for QuakeAlert

echo "🌊 Starting QuakeAlert..."

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Start FastAPI server
echo "Starting FastAPI server on http://127.0.0.1:8000"
echo "API docs available at http://127.0.0.1:8000/docs"
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

