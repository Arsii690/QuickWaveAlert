"""
Real-time Seismic Wave Signal Processing Module
Implements STA/LTA (Short Term Average / Long Term Average) algorithm
for P-wave detection in continuous seismic data streams.
"""

import numpy as np
from obspy import Stream, Trace, UTCDateTime
from typing import Tuple, List, Optional
import logging

logger = logging.getLogger(__name__)


class STALTADetector:
    """
    STA/LTA (Short Term Average / Long Term Average) detector for seismic events.
    
    This algorithm detects P-wave arrivals by comparing:
    - STA: Short-term average (recent vibration)
    - LTA: Long-term average (background noise)
    
    When STA/LTA ratio exceeds threshold, an earthquake is detected.
    """
    
    def __init__(
        self,
        sta_window: float = 1.0,  # Short-term window in seconds
        lta_window: float = 60.0,  # Long-term window in seconds
        threshold: float = 5.0,    # Detection threshold (STA/LTA ratio)
        overlap: float = 0.5       # Window overlap fraction
    ):
        """
        Initialize STA/LTA detector.
        
        Args:
            sta_window: Short-term averaging window (seconds)
            lta_window: Long-term averaging window (seconds)
            threshold: Detection threshold (STA/LTA must exceed this)
            overlap: Window overlap (0.0 to 1.0)
        """
        self.sta_window = sta_window
        self.lta_window = lta_window
        self.threshold = threshold
        self.overlap = overlap
        
    def compute_sta_lta(
        self, 
        data: np.ndarray, 
        sampling_rate: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute STA and LTA values for input waveform data.
        
        Args:
            data: Seismic waveform data (ground velocity)
            sampling_rate: Sampling rate in Hz
            
        Returns:
            Tuple of (STA array, LTA array, STA/LTA ratio array)
        """
        n_samples = len(data)
        sta_samples = int(self.sta_window * sampling_rate)
        lta_samples = int(self.lta_window * sampling_rate)
        
        # Ensure we have enough data
        if n_samples < lta_samples:
            logger.warning(f"Insufficient data: {n_samples} samples < {lta_samples} required")
            return (
                np.zeros(n_samples),
                np.ones(n_samples),
                np.zeros(n_samples)
            )
        
        # Compute squared data (energy)
        squared_data = data ** 2
        
        # Initialize arrays
        sta = np.zeros(n_samples)
        lta = np.ones(n_samples)  # Start with 1 to avoid division by zero
        
        # Compute STA (short-term average)
        for i in range(sta_samples, n_samples):
            sta[i] = np.mean(squared_data[i - sta_samples:i])
        
        # Compute LTA (long-term average)
        for i in range(lta_samples, n_samples):
            lta[i] = np.mean(squared_data[i - lta_samples:i])
        
        # Compute ratio (avoid division by zero)
        ratio = np.zeros(n_samples)
        mask = lta > 0
        ratio[mask] = sta[mask] / lta[mask]
        
        return sta, lta, ratio
    
    def detect_events(
        self,
        stream: Stream,
        min_duration: float = 2.0
    ) -> List[dict]:
        """
        Detect seismic events in an ObsPy Stream.
        
        Args:
            stream: ObsPy Stream containing seismic data
            min_duration: Minimum event duration in seconds
            
        Returns:
            List of detected events, each with:
            - start_time: Event start time
            - peak_time: Peak STA/LTA time
            - peak_ratio: Maximum STA/LTA ratio
            - station: Station code
            - channel: Channel code
        """
        events = []
        
        for trace in stream:
            data = trace.data
            sampling_rate = trace.stats.sampling_rate
            
            # Compute STA/LTA
            sta, lta, ratio = self.compute_sta_lta(data, sampling_rate)
            
            # Find peaks above threshold
            above_threshold = ratio > self.threshold
            
            if not np.any(above_threshold):
                continue
            
            # Find continuous regions above threshold
            diff = np.diff(above_threshold.astype(int))
            starts = np.where(diff == 1)[0] + 1
            ends = np.where(diff == -1)[0] + 1
            
            # Handle edge cases
            if above_threshold[0]:
                starts = np.concatenate([[0], starts])
            if above_threshold[-1]:
                ends = np.concatenate([ends, [len(above_threshold)]])
            
            # Process each detected region
            for start_idx, end_idx in zip(starts, ends):
                duration = (end_idx - start_idx) / sampling_rate
                
                if duration < min_duration:
                    continue
                
                # Find peak in this region
                region_ratio = ratio[start_idx:end_idx]
                peak_idx = start_idx + np.argmax(region_ratio)
                peak_ratio = ratio[peak_idx]
                
                # Convert to time
                start_time = trace.stats.starttime + start_idx / sampling_rate
                peak_time = trace.stats.starttime + peak_idx / sampling_rate
                
                events.append({
                    'start_time': start_time.isoformat(),
                    'peak_time': peak_time.isoformat(),
                    'peak_ratio': float(peak_ratio),
                    'station': trace.stats.station,
                    'channel': trace.stats.channel,
                    'network': trace.stats.network,
                    'location': trace.stats.location,
                    'duration': duration,
                    'sampling_rate': sampling_rate
                })
        
        return events
    
    def extract_features(
        self,
        stream: Stream,
        event: dict
    ) -> dict:
        """
        Extract features from a detected seismic event.
        
        Args:
            stream: ObsPy Stream containing the event
            event: Event dictionary from detect_events()
            
        Returns:
            Dictionary of extracted features
        """
        features = {}
        
        # Find the trace for this event
        for trace in stream:
            if (trace.stats.station == event['station'] and 
                trace.stats.channel == event['channel']):
                
                # Get event window (convert ISO string to UTCDateTime)
                event_start = UTCDateTime(event['start_time'])
                event_end = event_start + event['duration']
                
                # Extract event data
                event_trace = trace.slice(event_start, event_end)
                data = event_trace.data
                
                # Compute features
                features['peak_amplitude'] = float(np.max(np.abs(data)))
                features['rms_amplitude'] = float(np.sqrt(np.mean(data ** 2)))
                features['mean_amplitude'] = float(np.mean(np.abs(data)))
                features['std_amplitude'] = float(np.std(data))
                features['peak_ratio'] = event['peak_ratio']
                features['duration'] = event['duration']
                features['sampling_rate'] = event['sampling_rate']
                
                # Frequency domain features
                fft = np.fft.rfft(data)
                freqs = np.fft.rfftfreq(len(data), 1.0 / event['sampling_rate'])
                power = np.abs(fft) ** 2
                
                # Dominant frequency
                dominant_freq_idx = np.argmax(power)
                features['dominant_frequency'] = float(freqs[dominant_freq_idx])
                
                # Spectral centroid
                if np.sum(power) > 0:
                    features['spectral_centroid'] = float(
                        np.sum(freqs * power) / np.sum(power)
                    )
                else:
                    features['spectral_centroid'] = 0.0
                
                break
        
        return features

