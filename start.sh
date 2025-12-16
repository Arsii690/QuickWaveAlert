#!/bin/bash
# Startup script for local development and deployment

echo "🌊 Starting QuakeAlert..."

# Check if virtual environment exists
if [ -d "venv" ]; then
    echo "📦 Activating virtual environment..."
    source venv/bin/activate
fi

# Start FastAPI in background
echo "🚀 Starting FastAPI server on port 8000..."
uvicorn app.main:app --host 0.0.0.0 --port 8000 --log-level info &
FASTAPI_PID=$!

# Wait for FastAPI to start
sleep 3

# Start Streamlit
echo "📊 Starting Streamlit dashboard on port 8501..."
streamlit run dashboard.py --server.port 8501 --server.address 0.0.0.0 --server.headless true &
STREAMLIT_PID=$!

echo ""
echo "============================================="
echo "🌊 QuakeAlert is running!"
echo "============================================="
echo "📡 FastAPI:   http://localhost:8000"
echo "📊 Dashboard: http://localhost:8501"
echo "📚 API Docs:  http://localhost:8000/docs"
echo "============================================="
echo ""
echo "Press Ctrl+C to stop all services"

# Wait for both processes
wait $FASTAPI_PID $STREAMLIT_PID

