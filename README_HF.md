---
title: QuakeAlert
emoji: 🌊
colorFrom: blue
colorTo: purple
sdk: docker
sdk_version: latest
app_port: 7860
pinned: false
license: mit
---

# QuakeAlert - Real-time Seismic Wave Analysis

Real-time earthquake detection using live seismic waveform data and STA/LTA signal processing.

## Features

- 🌊 Real-time P-wave detection from FDSN seismic streams
- 📡 Supports multiple seismic stations (IRIS, USGS)
- 🔍 STA/LTA algorithm for accurate event detection
- 📊 Interactive dashboard with visualizations
- ☁️ MLOps integration with Hopsworks

## Usage

1. Select a seismic station from the dropdown
2. Choose analysis duration (1-60 minutes)
3. Click "Analyze Station" to detect earthquakes
4. View detected events with detailed metrics

## API Endpoints

- `POST /analyze_waveform` - Analyze waveform from a station
- `POST /stations` - Get available stations
- `GET /` - Health check

## Technology

- FastAPI backend
- ObsPy for seismic data
- STA/LTA signal processing
- Streamlit dashboard

