"""
Historical Data Backfill Script
Loads historical seismic data, processes it, and uploads to Hopsworks Feature Store.

CONFIGURATION:
- DAYS_TO_BACKFILL = 15 days
- BATCH_SIZE_HOURS = 12 hours per batch
- TOTAL BATCHES = 15 days × 2 batches/day = 30 batches
"""

import os
import hopsworks
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from pathlib import Path
import logging
import time
from typing import List, Dict, Optional
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from app.signal_processor import STALTADetector
from app.data_fetcher import SeismicDataFetcher
from obspy import UTCDateTime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(message)s'
)
logger = logging.getLogger(__name__)

# Load Configuration
env_path = Path(__file__).parent / '.env'
load_dotenv(dotenv_path=env_path)

API_KEY = os.getenv("HOPSWORKS_API_KEY")
PROJECT_NAME = os.getenv("HOPSWORKS_PROJECT_NAME", "QuakeAlertWave")

# Initialize components
detector = STALTADetector(sta_window=1.0, lta_window=60.0, threshold=5.0)
data_fetcher = SeismicDataFetcher(client_url="IRIS")

# Monitoring stations (4 stations)
MONITORING_STATIONS = [
    ("IU", "ANMO"),  # Albuquerque, New Mexico
    ("IU", "COLA"),  # College, Alaska
    ("US", "TAU"),   # Tucson, Arizona
    ("IU", "MAJO"),  # Matsushiro, Japan
]

# ============================================================
# CONFIGURATION - EASY TO UNDERSTAND
# ============================================================
DAYS_TO_BACKFILL = 15           # Process 15 days of data
BATCH_SIZE_HOURS = 12           # Each batch = 12 hours

# CALCULATION:
# - 24 hours/day ÷ 12 hours/batch = 2 batches per day
# - 15 days × 2 batches = 30 TOTAL BATCHES
BATCHES_PER_DAY = 24 // BATCH_SIZE_HOURS  # = 2
TOTAL_BATCHES = DAYS_TO_BACKFILL * BATCHES_PER_DAY  # = 30
# ============================================================


def process_time_window(network: str, station: str, starttime: UTCDateTime, endtime: UTCDateTime) -> List[Dict]:
    """Process a single time window for a station and extract events."""
    events = []
    
    try:
        stream = data_fetcher.fetch_waveform(
            network=network,
            station=station,
            starttime=starttime,
            endtime=endtime,
            location="*",
            channel="BHZ"
        )
        
        if stream and len(stream) > 0:
            detected_events = detector.detect_events(stream)
            
            for event in detected_events:
                try:
                    features = detector.extract_features(stream, event)
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
                    events.append(record)
                except Exception:
                    continue
                    
    except Exception as e:
        logger.warning(f"  ⚠️ {network}.{station}: {str(e)[:50]}")
    
    return events


def process_features(df: pd.DataFrame) -> pd.DataFrame:
    """Process and validate extracted features."""
    if df.empty:
        return df
    
    # Note: All features from extract_features() are kept:
    # - peak_amplitude, rms_amplitude, mean_amplitude, std_amplitude
    # - peak_ratio, duration, sampling_rate
    # - dominant_frequency, spectral_centroid
    # The Feature Group schema will auto-update on first insert if new columns are present
    
    # Add derived features
    df['is_significant'] = (df['peak_ratio'] > 10.0).astype(int)
    df['magnitude_estimate'] = df['peak_amplitude'].apply(
        lambda x: 0.0 if x == 0 or pd.isna(x) else min(9.0, 2.0 + 2.0 * np.log10(abs(x)))
    )
    
    # Convert timestamp for Hopsworks
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df['timestamp'] = df['timestamp'].astype('int64') // 10**6
    
    df = df.fillna(0)
    return df


def get_or_create_feature_group(fs):
    """Get existing feature group or create a new one."""
    try:
        wave_fg = fs.get_feature_group(name="waveform_features", version=1)
        if wave_fg is not None and hasattr(wave_fg, 'insert'):
            logger.info("✅ Using existing feature group 'waveform_features'")
            return wave_fg
    except Exception:
        pass
    
    # Create new feature group
    logger.info("📦 Creating new feature group 'waveform_features'...")
    try:
        wave_fg = fs.create_feature_group(
            name="waveform_features",
            version=1,
            description="Seismic waveform event features from STA/LTA detection",
            primary_key=["timestamp", "network", "station"],
            event_time="timestamp"
        )
        if wave_fg is not None:
            logger.info("✅ Feature group created successfully")
            return wave_fg
    except Exception as e:
        logger.error(f"❌ Failed to create feature group: {e}")
    
    return None


def upload_to_hopsworks(df: pd.DataFrame, wave_fg) -> int:
    """Upload data to Hopsworks with retry logic for Kafka timeouts and connection errors.
    Returns rows uploaded."""
    if df.empty or wave_fg is None:
        return 0
    
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
        return len(df)
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
                df_fixed = df.drop(columns=[col for col in problematic_cols if col in df.columns])
                # Now try uploading fixed dataframe with retry logic
                df = df_fixed
            else:
                # Schema error but couldn't identify columns - not retryable
                logger.error(f"❌ Upload failed (schema error): {e}")
                return 0
    
    # Retry logic for Kafka timeouts and connection errors
    for attempt in range(MAX_RETRIES):
        try:
            wave_fg.insert(df)
            if attempt > 0:
                logger.info(f"   ✅ Upload successful after {attempt + 1} attempts")
            return len(df)
        except Exception as e:
            error_msg = str(e)
            
            # Check if error is retryable
            if is_retryable_error(error_msg):
                if attempt < MAX_RETRIES - 1:
                    # Calculate exponential backoff delay
                    delay = min(INITIAL_RETRY_DELAY * (2 ** attempt), MAX_RETRY_DELAY)
                    logger.warning(f"   ⚠️ Upload failed (attempt {attempt + 1}/{MAX_RETRIES}): {error_msg[:100]}")
                    logger.info(f"   🔄 Retrying in {delay} seconds...")
                    time.sleep(delay)
                    continue
                else:
                    # Final attempt failed
                    logger.error(f"❌ Upload failed after {MAX_RETRIES} attempts: {error_msg[:200]}")
                    return 0
            else:
                # Non-retryable error (e.g., schema, validation)
                logger.error(f"❌ Upload failed (non-retryable): {error_msg[:200]}")
                return 0
    
    return 0


def backfill_historical_data():
    """Main backfill function - processes 15 days of data in 30 batches."""
    
    # Header
    logger.info("=" * 60)
    logger.info("🌊 QUAKEALERT HISTORICAL DATA BACKFILL")
    logger.info("=" * 60)
    logger.info(f"📊 Configuration:")
    logger.info(f"   Days to process: {DAYS_TO_BACKFILL}")
    logger.info(f"   Batch size: {BATCH_SIZE_HOURS} hours")
    logger.info(f"   Batches per day: {BATCHES_PER_DAY}")
    logger.info(f"   ⭐ TOTAL BATCHES: {TOTAL_BATCHES}")
    logger.info(f"   Stations: {len(MONITORING_STATIONS)}")
    logger.info("=" * 60)
    
    # Check API key
    if not API_KEY:
        logger.error("❌ HOPSWORKS_API_KEY not found in .env file")
        return
    
    # Connect to Hopsworks
    logger.info("\n☁️ Connecting to Hopsworks...")
    try:
        project = hopsworks.login(api_key_value=API_KEY, project=PROJECT_NAME)
        fs = project.get_feature_store()
        logger.info(f"✅ Connected to project: {PROJECT_NAME}")
    except Exception as e:
        logger.error(f"❌ Failed to connect: {e}")
        return
    
    # Get or create feature group
    wave_fg = get_or_create_feature_group(fs)
    if wave_fg is None:
        logger.error("❌ Cannot proceed without feature group")
        return
    
    # Calculate time range (last N days from now)
    end_time = UTCDateTime.now()
    start_time = end_time - (DAYS_TO_BACKFILL * 24 * 60 * 60)
    
    logger.info(f"\n📅 Time Range:")
    logger.info(f"   From: {start_time.strftime('%Y-%m-%d %H:%M')}")
    logger.info(f"   To:   {end_time.strftime('%Y-%m-%d %H:%M')}")
    logger.info("=" * 60)
    
    # Progress tracking
    batch_number = 0
    total_events = 0
    total_uploaded = 0
    all_events = []
    start_run_time = time.time()
    
    # Process each day
    for day in range(DAYS_TO_BACKFILL):
        day_start = start_time + (day * 24 * 60 * 60)
        
        logger.info(f"\n{'─' * 60}")
        logger.info(f"📆 DAY {day + 1}/{DAYS_TO_BACKFILL}: {day_start.strftime('%Y-%m-%d')}")
        logger.info(f"{'─' * 60}")
        
        # Process each batch in the day (2 batches per day with 12-hour windows)
        for batch_in_day in range(BATCHES_PER_DAY):
            batch_number += 1
            
            batch_start = day_start + (batch_in_day * BATCH_SIZE_HOURS * 60 * 60)
            batch_end = batch_start + (BATCH_SIZE_HOURS * 60 * 60)
            
            # Progress bar
            progress = batch_number / TOTAL_BATCHES * 100
            logger.info(f"\n🔄 BATCH {batch_number}/{TOTAL_BATCHES} ({progress:.0f}%)")
            logger.info(f"   Time: {batch_start.strftime('%H:%M')} → {batch_end.strftime('%H:%M')}")
            
            batch_events = []
            
            # Process each station
            for network, station in MONITORING_STATIONS:
                events = process_time_window(network, station, batch_start, batch_end)
                batch_events.extend(events)
                if events:
                    logger.info(f"   ✓ {network}.{station}: {len(events)} events")
            
            all_events.extend(batch_events)
            total_events += len(batch_events)
            
            logger.info(f"   📊 Batch events: {len(batch_events)} | Total: {total_events}")
            
            # Upload every 500 events or at the end of each day
            if len(all_events) >= 500 or (batch_in_day == BATCHES_PER_DAY - 1):
                if all_events:
                    df = pd.DataFrame(all_events)
                    processed_df = process_features(df)
                    uploaded = upload_to_hopsworks(processed_df, wave_fg)
                    total_uploaded += uploaded
                    logger.info(f"   ☁️ Uploaded: {uploaded} rows (Total: {total_uploaded})")
                    all_events = []
    
    # Upload any remaining events
    if all_events:
        df = pd.DataFrame(all_events)
        processed_df = process_features(df)
        uploaded = upload_to_hopsworks(processed_df, wave_fg)
        total_uploaded += uploaded
        logger.info(f"\n☁️ Final upload: {uploaded} rows")
    
    # Summary
    elapsed = time.time() - start_run_time
    logger.info(f"\n{'=' * 60}")
    logger.info("🎉 BACKFILL COMPLETE!")
    logger.info(f"{'=' * 60}")
    logger.info(f"📊 Results:")
    logger.info(f"   Batches completed: {batch_number}/{TOTAL_BATCHES}")
    logger.info(f"   Events detected: {total_events}")
    logger.info(f"   Rows uploaded: {total_uploaded}")
    logger.info(f"   Time elapsed: {elapsed/60:.1f} minutes")
    logger.info(f"{'=' * 60}\n")


if __name__ == "__main__":
    try:
        backfill_historical_data()
    except KeyboardInterrupt:
        logger.info("\n\n⏹️ Backfill stopped by user. Data uploaded so far is saved!")
    except Exception as e:
        logger.error(f"\n❌ Fatal error: {e}")
