---
title: QuakeAlertWave
emoji: 🌊
colorFrom: blue
colorTo: purple
sdk: docker
sdk_version: latest
app_port: 7860
pinned: false
license: mit
---

# 🌊 QuakeAlert - Real-time Seismic Wave Analysis

**Advanced MLOps project for real-time earthquake detection using live seismic waveform data and STA/LTA signal processing algorithm.**

## 🎯 Project Overview

QuakeAlert is a production-ready MLOps system that:
- **Fetches live seismic data** from FDSN (Federation of Digital Seismograph Networks) via IRIS
- **Detects P-waves in real-time** using STA/LTA (Short Term Average / Long Term Average) algorithm
- **Extracts features** from detected events and stores them in Hopsworks Feature Store
- **Deploys as a scalable API** on Hugging Face Spaces with Docker
- **Implements CI/CD** with GitHub Actions

## 🏗️ Architecture

```
┌─────────────────┐
│  FDSN/IRIS      │  Live Seismic Waveform Data
│  Web Services   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Data Fetcher   │  ObsPy Client
│  (ObsPy)        │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Signal         │  STA/LTA Algorithm
│  Processor      │  P-Wave Detection
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Feature        │  Feature Extraction
│  Extractor      │  (Amplitude, Frequency, etc.)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Hopsworks      │  Feature Store
│  Feature Store  │  Model Registry
└─────────────────┘
         │
         ▼
┌─────────────────┐
│  FastAPI        │  REST API
│  Backend        │  /analyze_waveform
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Streamlit      │  Dashboard
│  Dashboard      │  Real-time Visualization
└─────────────────┘
```

## 🔬 How STA/LTA Detection Works

1. **Short Term Average (STA)**: Calculates average vibration over the last 1 second
2. **Long Term Average (LTA)**: Calculates average vibration over the last 60 seconds
3. **Detection Trigger**: When `STA / LTA > threshold` (default: 5.0), an earthquake is detected
4. **Feature Extraction**: For each detected event, extracts:
   - Peak amplitude, RMS amplitude
   - Dominant frequency, spectral centroid
   - Event duration and timing

## 📦 Installation

### Prerequisites

- Python 3.11+
- Docker (for containerized deployment)
- Hopsworks account (for feature store - optional)
- FDSN access (IRIS is public and free)

### Local Setup

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd QuakeAlert
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables** (optional, for Hopsworks)
   ```bash
   cp .env.example .env
   # Edit .env and add:
   # HOPSWORKS_API_KEY=your_key_here
   # HOPSWORKS_PROJECT_NAME=QuakeAlert
   ```

5. **Run the API**
   ```bash
   uvicorn app.main:app --reload
   ```

6. **Run the dashboard** (in another terminal)
   ```bash
   streamlit run dashboard.py
   ```

## 🚀 Usage

### API Endpoints

#### 1. Analyze Waveform
```bash
POST /analyze_waveform
{
  "network": "IU",
  "station": "ANMO",
  "location": "*",
  "channel": "BHZ",
  "duration_minutes": 10
}
```

**Response:**
```json
{
  "status": "success",
  "station": {
    "network": "IU",
    "station": "ANMO",
    "channel": "BHZ"
  },
  "analysis": {
    "events_detected": 2,
    "events": [
      {
        "start_time": "2025-01-15T10:30:00",
        "peak_ratio": 8.5,
        "peak_amplitude": 1.2e-5,
        "duration": 3.2,
        "dominant_frequency": 2.5
      }
    ]
  }
}
```

#### 2. Get Available Stations
```bash
POST /stations
{
  "network": "*",
  "station": "*",
  "minlatitude": -90.0,
  "maxlatitude": 90.0,
  "minlongitude": -180.0,
  "maxlongitude": 180.0
}
```

#### 3. Health Check
```bash
GET /
```

### Running Pipelines

#### Feature Pipeline (Prefect)
```bash
python pipelines/2_waveform_pipeline.py
```

This will:
- Fetch data from monitoring stations
- Detect events using STA/LTA
- Extract features
- Upload to Hopsworks Feature Store

## 🐳 Docker Deployment

### Build Image
```bash
docker build -t quakealert:latest .
```

### Run Container
```bash
docker run -p 8000:8000 \
  -e HOPSWORKS_API_KEY=your_key \
  -e HOPSWORKS_PROJECT_NAME=QuakeAlert \
  quakealert:latest
```

## 🤗 Hugging Face Spaces Deployment

### Prerequisites
1. Hugging Face account
2. Create a new Space (Docker SDK)
3. Add secrets: `HOPSWORKS_API_KEY`, `HOPSWORKS_PROJECT_NAME`

### Deploy
1. Push code to GitHub
2. Connect GitHub repo to Hugging Face Space
3. Space will auto-deploy on push to main branch

### Access Your Deployed Model
Once deployed, your Space will be available at:
```
https://huggingface.co/spaces/<your-username>/<space-name>
```

The API will be accessible at:
```
https://<your-username>-<space-name>.hf.space
```

## 🔄 CI/CD Pipeline

The project includes GitHub Actions workflow (`.github/workflows/ci-cd.yml`) that:

1. **Tests**: Runs linting and import tests
2. **Builds**: Creates Docker image
3. **Deploys**: Pushes to Hugging Face Spaces (on main branch)

### Setup CI/CD Secrets

In GitHub repository settings, add:
- `DOCKER_USERNAME` (optional, for Docker Hub)
- `DOCKER_PASSWORD` (optional)
- `HF_TOKEN` (Hugging Face token for deployment)

## 📊 Monitoring Stations

Default monitoring stations (configurable in `pipelines/2_waveform_pipeline.py`):
- **IU.ANMO**: Albuquerque, New Mexico
- **IU.COLA**: College, Alaska
- **US.TAU**: Tucson, Arizona
- **IU.MAJO**: Matsushiro, Japan

## 🛠️ Technology Stack

- **Backend**: FastAPI
- **Data Processing**: ObsPy, NumPy, SciPy
- **Signal Processing**: Custom STA/LTA implementation
- **MLOps**: Hopsworks (Feature Store, Model Registry)
- **Orchestration**: Prefect
- **Visualization**: Streamlit, Plotly
- **Deployment**: Docker, Hugging Face Spaces
- **CI/CD**: GitHub Actions

## 📝 Project Structure

```
QuakeAlert/
├── app/
│   ├── main.py              # FastAPI application
│   ├── signal_processor.py  # STA/LTA detector
│   └── data_fetcher.py      # FDSN data fetcher
├── pipelines/
│   ├── 2_waveform_pipeline.py  # Feature pipeline
│   └── 3_training_pipeline.py   # Model training (legacy)
├── dashboard.py              # Streamlit dashboard
├── Dockerfile               # Docker configuration
├── requirements.txt         # Python dependencies
├── .github/
│   └── workflows/
│       └── ci-cd.yml       # CI/CD pipeline
└── README.md               # This file
```

## 🎓 MLOps Features Demonstrated

✅ **Feature Store**: Hopsworks for storing waveform features  
✅ **Model Registry**: Hopsworks for model versioning  
✅ **Orchestration**: Prefect for pipeline management  
✅ **Containerization**: Docker for reproducible deployments  
✅ **CI/CD**: GitHub Actions for automated testing and deployment  
✅ **API Deployment**: Hugging Face Spaces for scalable hosting  
✅ **Monitoring**: Real-time dashboard with Streamlit  

## 🔧 Configuration

### STA/LTA Parameters

Adjust in `app/signal_processor.py`:
```python
detector = STALTADetector(
    sta_window=1.0,      # Short-term window (seconds)
    lta_window=60.0,     # Long-term window (seconds)
    threshold=5.0        # Detection threshold
)
```

### FDSN Client

Change data source in `app/data_fetcher.py`:
```python
data_fetcher = SeismicDataFetcher(client_url="IRIS")  # or "USGS"
```

## 🐛 Troubleshooting

### "No data available" error
- Check if station is online: Visit [IRIS Station Monitor](http://ds.iris.edu/gmap/)
- Verify network/station codes are correct
- Try different time windows

### ObsPy connection issues
- Check internet connection
- IRIS service may be temporarily unavailable
- Try different FDSN client (USGS, etc.)

### Hopsworks errors
- Verify API key in `.env`
- Check project name matches your Hopsworks project
- Feature group will be created automatically on first run

## 📚 References

- [ObsPy Documentation](https://docs.obspy.org/)
- [FDSN Web Services](https://www.fdsn.org/webservices/)
- [IRIS Data Services](https://service.iris.edu/)
- [STA/LTA Algorithm](https://en.wikipedia.org/wiki/STA/LTA)

## 📄 License

This project is for educational purposes as part of an MLOps course.

## 👥 Author

Built as a semester project demonstrating real-world MLOps practices.

---

**🌊 Real-time Earthquake Detection | Powered by STA/LTA Signal Processing**

