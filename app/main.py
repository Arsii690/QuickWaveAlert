"""
QuakeAlert API - Real-time Seismic Wave Analysis
Uses ObsPy and STA/LTA algorithm for P-wave detection from live seismic streams.
"""

import os
import joblib
import hopsworks
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field
from dotenv import load_dotenv
from pathlib import Path
from typing import Optional, List
from datetime import datetime
import logging

from app.signal_processor import STALTADetector
from app.data_fetcher import SeismicDataFetcher
from obspy import UTCDateTime, Stream

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 1. Initialize API
app = FastAPI(
    title="QuakeAlert API - Real-time Wave Analysis",
    version="2.0",
    description="Real-time seismic P-wave detection using STA/LTA algorithm"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. Load Env & Config
API_KEY = None
PROJECT_NAME = None

try:
    env_path = Path(__file__).parent.parent / '.env'
    load_dotenv(dotenv_path=env_path)
    API_KEY = os.getenv("HOPSWORKS_API_KEY")
    PROJECT_NAME = os.getenv("HOPSWORKS_PROJECT_NAME", "QuakeAlertWave")
except Exception as e:
    logger.warning(f"Environment loading issue: {e}")

# 3. Global Components
detector = STALTADetector(
    sta_window=1.0,      # 1 second short-term window
    lta_window=60.0,     # 60 second long-term window
    threshold=5.0        # Detection threshold
)
data_fetcher = SeismicDataFetcher(client_url="IRIS")

# 4. Input Schemas
class WaveformAnalysisRequest(BaseModel):
    """Request for waveform analysis from station data."""
    network: str = Field(..., example="IU", description="Network code (e.g., IU, US)")
    station: str = Field(..., example="ANMO", description="Station code")
    location: str = Field("*", example="*", description="Location code")
    channel: str = Field("BHZ", example="BHZ", description="Channel code")
    duration_minutes: int = Field(10, ge=1, le=60, description="Minutes of data to analyze")

class StationInfoRequest(BaseModel):
    """Request for station information."""
    network: Optional[str] = "*"
    station: Optional[str] = "*"
    minlatitude: float = -90.0
    maxlatitude: float = 90.0
    minlongitude: float = -180.0
    maxlongitude: float = 180.0

class QuakeInput(BaseModel):
    """Legacy input schema for backward compatibility."""
    latitude: float
    longitude: float
    depth: float

    model_config = ConfigDict(json_schema_extra={
        "example": {
            "latitude": 38.3,
            "longitude": 142.4,
            "depth": 10.0
        }
    })

# 5. Startup: Initialize Components
@app.on_event("startup")
def startup():
    """Initialize components on startup."""
    logger.info("🚀 QuakeAlert API starting up...")
    logger.info("📡 Initialized FDSN client (IRIS)")
    logger.info("🔍 Initialized STA/LTA detector")
    
    # Try to load models from Hopsworks (optional, for legacy support)
    if API_KEY:
        try:
            project = hopsworks.login(api_key_value=API_KEY, project=PROJECT_NAME)
            logger.info("✅ Connected to Hopsworks")
        except Exception as e:
            logger.warning(f"⚠️ Hopsworks connection failed: {e}")

# 6. Health Check
@app.get("/")
def health():
    """Health check endpoint."""
    return {
        "status": "running",
        "service": "QuakeAlert Real-time Wave Analysis",
        "version": "2.0",
        "detector": "STA/LTA",
        "data_source": "FDSN (IRIS)"
    }

# 7. Real-time Wave Analysis Endpoint
@app.post("/analyze_waveform")
async def analyze_waveform(request: WaveformAnalysisRequest):
    """
    Analyze seismic waveform data from a station for P-wave detection.
    
    Fetches live data from FDSN and applies STA/LTA algorithm to detect earthquakes.
    """
    try:
        logger.info(
            f"Fetching waveform from {request.network}.{request.station} "
            f"({request.duration_minutes} minutes)"
        )
        
        # Fetch waveform data
        stream = data_fetcher.fetch_recent_data(
            network=request.network,
            station=request.station,
            duration_minutes=request.duration_minutes,
            location=request.location,
            channel=request.channel
        )
        
        if stream is None or len(stream) == 0:
            raise HTTPException(
                status_code=404,
                detail=f"No data available for {request.network}.{request.station} with channel {request.channel}. Try a different channel (BHZ, HHZ, EHZ) or check if the station has data for this channel."
            )
        
        # Detect events
        events = detector.detect_events(stream)
        
        # Extract features for each event
        event_features = []
        for event in events:
            features = detector.extract_features(stream, event)
            event_features.append({
                **event,
                **features
            })
        
        return {
            "status": "success",
            "station": {
                "network": request.network,
                "station": request.station,
                "channel": request.channel
            },
            "analysis": {
                "duration_minutes": request.duration_minutes,
                "events_detected": len(events),
                "events": event_features
            },
            "detection_params": {
                "sta_window": detector.sta_window,
                "lta_window": detector.lta_window,
                "threshold": detector.threshold
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# 8. Get Available Stations
@app.post("/stations")
async def get_stations(request: StationInfoRequest):
    """Get list of available seismic stations."""
    try:
        stations = data_fetcher.get_stations(
            network=request.network or "*",
            station=request.station or "*",
            minlatitude=request.minlatitude,
            maxlatitude=request.maxlatitude,
            minlongitude=request.minlongitude,
            maxlongitude=request.maxlongitude
        )
        
        return {
            "status": "success",
            "count": len(stations),
            "stations": stations
        }
        
    except Exception as e:
        logger.error(f"Station fetch error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# 9. Legacy Prediction Endpoint (for backward compatibility)
@app.post("/predict")
def predict(data: QuakeInput):
    """
    Legacy endpoint for magnitude prediction (backward compatibility).
    Note: This requires models to be loaded from Hopsworks.
    """
    return {
        "status": "deprecated",
        "message": "This endpoint is deprecated. Use /analyze_waveform for real-time wave analysis.",
        "suggestion": "Use /analyze_waveform endpoint with station information"
    }

# 10. Detection Statistics
@app.get("/detector/info")
def detector_info():
    """Get information about the STA/LTA detector configuration."""
    return {
        "algorithm": "STA/LTA (Short Term Average / Long Term Average)",
        "parameters": {
            "sta_window_seconds": detector.sta_window,
            "lta_window_seconds": detector.lta_window,
            "threshold": detector.threshold,
            "overlap": detector.overlap
        },
        "description": (
            "Detects P-wave arrivals by comparing recent vibration (STA) "
            "to background noise (LTA). When STA/LTA exceeds threshold, "
            "an earthquake is detected."
        )
    }

