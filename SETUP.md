# 🚀 Local Setup Guide

This guide will help you set up the QuakeAlert project in a virtual environment on your local machine.

## Prerequisites

- Python 3.11 or higher
- pip (Python package installer)

## Step 1: Create Virtual Environment

### On macOS/Linux:
```bash
# Navigate to project directory
cd /Users/kainat/Desktop/QuakeAlertWave

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate
```

### On Windows:
```bash
# Navigate to project directory
cd C:\path\to\QuakeAlertWave

# Create virtual environment
python -m venv venv

# Activate virtual environment
venv\Scripts\activate
```

**Note:** Once activated, you'll see `(venv)` at the beginning of your terminal prompt.

## Step 2: Install Dependencies

```bash
# Make sure virtual environment is activated
pip install --upgrade pip
pip install -r requirements.txt
```

## Step 3: Configure Environment Variables

1. **Copy the example environment file:**
   ```bash
   cp .env.example .env
   ```
   
   (If you're on Windows and don't have `cp`, just copy `.env.example` and rename it to `.env`)

2. **Edit the `.env` file** and add your Hopsworks API key:
   ```bash
   # Open .env in your text editor
   # Replace 'your_hopsworks_api_key_here' with your actual API key
   HOPSWORKS_API_KEY=your_actual_api_key_here
   HOPSWORKS_PROJECT_NAME=QuakeAlert
   ```

3. **Get your Hopsworks API Key:**
   - Go to https://www.hopsworks.ai/
   - Sign up or log in
   - Create a project (or use existing)
   - Go to Settings → API Keys
   - Generate a new API key
   - Copy and paste it into your `.env` file

## Step 4: Verify Installation

```bash
# Test that imports work
python -c "from app.signal_processor import STALTADetector; print('✅ Signal processor OK')"
python -c "from app.data_fetcher import SeismicDataFetcher; print('✅ Data fetcher OK')"
python -c "from app.main import app; print('✅ FastAPI app OK')"
```

## Step 5: Run the Application

### Start the API Server:
```bash
uvicorn app.main:app --reload
```

The API will be available at: `http://127.0.0.1:8000`

### Start the Dashboard (in a new terminal):
```bash
# Make sure to activate venv in this terminal too
source venv/bin/activate  # On macOS/Linux
# or
venv\Scripts\activate  # On Windows

streamlit run dashboard.py
```

The dashboard will be available at: `http://localhost:8501`

## Deactivating Virtual Environment

When you're done working, you can deactivate the virtual environment:
```bash
deactivate
```

## Troubleshooting

### Virtual environment not activating?
- Make sure you're using the correct path
- On Windows, try: `.\venv\Scripts\activate`

### Import errors?
- Make sure virtual environment is activated
- Reinstall dependencies: `pip install -r requirements.txt`

### ObsPy installation issues?
- On macOS: `brew install gcc`
- On Ubuntu: `sudo apt-get install gcc python3-dev`

### Port already in use?
- Change the port: `uvicorn app.main:app --port 8001`
- Or stop the process using port 8000

## Next Steps

- Test the API endpoints
- Run the feature pipeline: `python pipelines/2_waveform_pipeline.py`
- Check the dashboard for real-time visualizations

