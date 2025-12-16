# 📁 Project Separation Summary

## Two Separate Projects

You now have **two independent projects**:

### 1. **QuakeAlert** (Original - `/Users/kainat/Desktop/QuakeAlert/`)
   - **Purpose**: CSV-based earthquake prediction using ML models
   - **Input**: Latitude, Longitude, Depth
   - **Output**: Magnitude, Tsunami Risk, Seismic Zone
   - **Technology**: FastAPI, Hopsworks, Prefect, scikit-learn
   - **Status**: ✅ **Preserved and untouched**

### 2. **QuakeAlertWave** (New - `/Users/kainat/Desktop/QuakeAlertWave/`)
   - **Purpose**: Real-time seismic wave analysis using STA/LTA
   - **Input**: Live seismic waveform data from FDSN/IRIS
   - **Output**: Detected P-waves, event features, real-time alerts
   - **Technology**: FastAPI, ObsPy, STA/LTA algorithm, Hopsworks, Prefect
   - **Status**: ✅ **New project ready to use**

## QuakeAlertWave Structure

```
QuakeAlertWave/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI with wave analysis endpoints
│   ├── signal_processor.py  # STA/LTA detector
│   └── data_fetcher.py      # FDSN data fetcher
├── pipelines/
│   └── 2_waveform_pipeline.py  # Feature extraction pipeline
├── dashboard.py             # Streamlit dashboard for wave analysis
├── app.py                   # Hugging Face entry point
├── Dockerfile               # Docker configuration
├── requirements.txt         # Dependencies (includes ObsPy)
├── README.md               # Full documentation
├── README_HF.md           # Hugging Face Space config
├── DEPLOYMENT.md          # Deployment guide
├── .github/workflows/
│   └── ci-cd.yml          # CI/CD pipeline
└── .dockerignore          # Docker ignore file
```

## Quick Start - QuakeAlertWave

### Local Development
```bash
cd /Users/kainat/Desktop/QuakeAlertWave

# Install dependencies
pip install -r requirements.txt

# Start API
uvicorn app.main:app --reload

# Start Dashboard (separate terminal)
streamlit run dashboard.py
```

### Test the API
```bash
# Health check
curl http://127.0.0.1:8000/

# Analyze waveform from a station
curl -X POST http://127.0.0.1:8000/analyze_waveform \
  -H "Content-Type: application/json" \
  -d '{
    "network": "IU",
    "station": "ANMO",
    "duration_minutes": 10
  }'
```

## Key Differences

| Feature | QuakeAlert (Original) | QuakeAlertWave (New) |
|---------|----------------------|---------------------|
| **Data Source** | CSV files (USGS) | Live FDSN streams (IRIS) |
| **Input** | lat/lon/depth | Network/Station codes |
| **Algorithm** | ML models (RF) | Signal processing (STA/LTA) |
| **Detection** | Prediction | Real-time detection |
| **Dependencies** | Basic ML stack | ObsPy, signal processing |
| **Use Case** | Historical analysis | Real-time monitoring |

## Next Steps

1. **Test QuakeAlertWave locally**
   - Run the API and dashboard
   - Test with different stations

2. **Deploy to Hugging Face**
   - Create a new Space
   - Push code to GitHub
   - Connect to Space

3. **Run Feature Pipeline**
   ```bash
   python pipelines/2_waveform_pipeline.py
   ```

## Notes

- Both projects are **completely independent**
- Original QuakeAlert is **untouched** and works as before
- QuakeAlertWave is **ready to use** with all new features
- You can work on either project without affecting the other

