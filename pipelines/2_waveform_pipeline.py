"""
Real-time Waveform Feature Pipeline
Fetches live seismic data, detects events using STA/LTA, and stores features in Hopsworks.
"""

import os
import hopsworks
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from prefect import task, flow, get_run_logger
from pathlib import Path
from datetime import datetime, timedelta
import time
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from app.signal_processor import STALTADetector
from app.data_fetcher import SeismicDataFetcher
from obspy import UTCDateTime

# Load Configuration
env_path = Path(__file__).parent.parent / '.env'
load_dotenv(dotenv_path=env_path)

API_KEY = os.getenv("HOPSWORKS_API_KEY")
PROJECT_NAME = os.getenv("HOPSWORKS_PROJECT_NAME", "QuakeAlert")

# Initialize components
detector = STALTADetector(sta_window=1.0, lta_window=60.0, threshold=5.0)
data_fetcher = SeismicDataFetcher(client_url="IRIS")

# Popular seismic stations for monitoring
MONITORING_STATIONS = [
    ("IU", "ANMO"),  # Albuquerque, New Mexico
    ("IU", "COLA"),  # College, Alaska
    ("US", "TAU"),   # Tucson, Arizona
    ("IU", "MAJO"),  # Matsushiro, Japan
]

@task(name="Fetch Waveform Data", retries=2, retry_delay_seconds=30)
def fetch_waveform_data():
    """Fetches recent waveform data from monitoring stations."""
    logger = get_run_logger()
    logger.info("📡 Fetching waveform data from monitoring stations...")
    
    all_events = []
    endtime = UTCDateTime.now()
    starttime = endtime - (10 * 60)  # Last 10 minutes
    
    for network, station in MONITORING_STATIONS:
        try:
            logger.info(f"Fetching from {network}.{station}...")
            stream = data_fetcher.fetch_waveform(
                network=network,
                station=station,
                channel="BHZ",
                starttime=starttime,
                endtime=endtime
            )
            
            if stream and len(stream) > 0:
                events = detector.detect_events(stream)
                
                for event in events:
                    # Extract features
                    features = detector.extract_features(stream, event)
                    
                    # Combine event info and features
                    record = {
                        'timestamp': int(UTCDateTime(event['start_time']).timestamp * 1000),
                        'network': event['network'],
                        'station': event['station'],
                        'channel': event['channel'],
                        'location': event.get('location', ''),
                        'peak_ratio': event['peak_ratio'],
                        'duration': event['duration'],
                        **features
                    }
                    all_events.append(record)
                    
                logger.info(f"  Found {len(events)} events at {network}.{station}")
            else:
                logger.warning(f"  No data from {network}.{station}")
                
        except Exception as e:
            logger.error(f"  Error fetching from {network}.{station}: {e}")
            continue
    
    logger.info(f"Total events detected: {len(all_events)}")
    return pd.DataFrame(all_events)

@task(name="Process Features")
def process_features(df):
    """Process and validate extracted features."""
    logger = get_run_logger()
    
    if df.empty:
        logger.warning("No events to process")
        return df
    
    # Note: All features from extract_features() are kept:
    # - peak_amplitude, rms_amplitude, mean_amplitude, std_amplitude
    # - peak_ratio, duration, sampling_rate
    # - dominant_frequency, spectral_centroid
    # The Feature Group schema will auto-update on first insert if new columns are present
    
    # Add derived features
    df['is_significant'] = (df['peak_ratio'] > 10.0).astype(int)
    df['magnitude_estimate'] = df['peak_amplitude'].apply(
        lambda x: 0.0 if x == 0 else min(9.0, 2.0 + 2.0 * np.log10(x))
    )
    
    # Ensure timestamp is in milliseconds
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df['timestamp'] = df['timestamp'].astype('int64') // 10**6
    
    logger.info(f"Processed {len(df)} event records")
    return df

@task(name="Upload to Feature Store")
def upload_to_hopsworks(df):
    """Upload processed features to Hopsworks Feature Store."""
    logger = get_run_logger()
    
    if df.empty:
        logger.warning("Skipping upload (no data)")
        return
    
    if not API_KEY:
        logger.warning("Hopsworks API key not found, skipping upload")
        return
    
    try:
        logger.info("☁️ Connecting to Hopsworks Feature Store...")
        project = hopsworks.login(api_key_value=API_KEY, project=PROJECT_NAME)
        fs = project.get_feature_store()
        
        # Get or create feature group
        wave_fg = None
        try:
            wave_fg = fs.get_feature_group(name="waveform_features", version=1)
            logger.info("Using existing feature group")
        except Exception as e:
            logger.info(f"Feature group not found: {e}")
            logger.info("Creating new feature group...")
            try:
            wave_fg = fs.create_feature_group(
                name="waveform_features",
                version=1,
                description="Real-time seismic waveform event features from STA/LTA detection",
                primary_key=["timestamp", "network", "station"],
                event_time="timestamp"
            )
                logger.info("Feature group created successfully")
            except Exception as e2:
                logger.error(f"Failed to create feature group: {e2}")
                raise
        
        if wave_fg is None:
            raise ValueError("Feature group is None, cannot proceed")
        
        logger.info(f"Inserting {len(df)} rows into Feature Group...")
        
        # Retry configuration
        MAX_RETRIES = 3
        INITIAL_RETRY_DELAY = 5  # seconds
        MAX_RETRY_DELAY = 30  # seconds
        
        # Check if error is retryable (Kafka timeout, connection errors)
        def is_retryable_error(error_msg: str) -> bool:
            """Check if error is retryable (network/Kafka issues)."""
            retryable_keywords = [
                'kafka',
                'timed out',
                'timeout',
                'connection refused',
                'connection error',
                'max retries exceeded',
                'failed to establish',
                'network',
                'temporarily unavailable'
            ]
            error_lower = error_msg.lower()
            return any(keyword in error_lower for keyword in retryable_keywords)
        
        # First, handle schema mismatch (non-retryable)
        try:
        wave_fg.insert(df)
        logger.info("✅ Upload successful!")
        except Exception as e:
            error_msg = str(e)
            
            # If schema mismatch, try removing the problematic columns
            if "does not exist in feature group" in error_msg or "not compatible" in error_msg.lower():
                logger.warning("⚠️ Schema mismatch detected. Attempting to fix...")
                problematic_cols = []
                if 'mean_amplitude' in error_msg:
                    problematic_cols.append('mean_amplitude')
                if 'std_amplitude' in error_msg:
                    problematic_cols.append('std_amplitude')
                if 'sampling_rate' in error_msg:
                    problematic_cols.append('sampling_rate')
                
                if problematic_cols:
                    logger.info(f"   Removing columns not in schema: {problematic_cols}")
                    df = df.drop(columns=[col for col in problematic_cols if col in df.columns])
                else:
                    logger.error(f"Upload failed (schema error): {e}")
                    raise
        
        # Retry logic for Kafka timeouts and connection errors
        for attempt in range(MAX_RETRIES):
            try:
                wave_fg.insert(df)
                if attempt > 0:
                    logger.info(f"✅ Upload successful after {attempt + 1} attempts")
                else:
                    logger.info("✅ Upload successful!")
                break
            except Exception as e:
                error_msg = str(e)
                
                # Check if error is retryable
                if is_retryable_error(error_msg):
                    if attempt < MAX_RETRIES - 1:
                        # Calculate exponential backoff delay
                        delay = min(INITIAL_RETRY_DELAY * (2 ** attempt), MAX_RETRY_DELAY)
                        logger.warning(f"⚠️ Upload failed (attempt {attempt + 1}/{MAX_RETRIES}): {error_msg[:100]}")
                        logger.info(f"🔄 Retrying in {delay} seconds...")
                        time.sleep(delay)
                        continue
                    else:
                        # Final attempt failed
                        logger.error(f"Upload failed after {MAX_RETRIES} attempts: {error_msg[:200]}")
                        raise
                else:
                    # Non-retryable error (e.g., schema, validation)
                    logger.error(f"Upload failed (non-retryable): {error_msg[:200]}")
                    raise
        
    except Exception as e:
        logger.error(f"Upload failed: {e}")

@flow(name="Waveform Feature Pipeline")
def waveform_pipeline():
    """Main orchestrator flow for waveform feature extraction."""
    df = fetch_waveform_data()
    processed_df = process_features(df)
    upload_to_hopsworks(processed_df)

if __name__ == "__main__":
    waveform_pipeline()

