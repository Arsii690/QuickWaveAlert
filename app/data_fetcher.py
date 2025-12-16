"""
Seismic Data Fetcher Module
Fetches live seismic waveform data from FDSN Web Services using ObsPy.
"""

from obspy import UTCDateTime, Stream
from obspy.clients.fdsn import Client
from typing import Optional, List, Tuple
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class SeismicDataFetcher:
    """
    Fetches seismic waveform data from FDSN-compatible web services.
    """
    
    def __init__(
        self,
        client_url: str = "IRIS",  # IRIS, USGS, or custom FDSN URL
        timeout: int = 30
    ):
        """
        Initialize FDSN client.
        
        Args:
            client_url: FDSN service URL or "IRIS"/"USGS" for default services
            timeout: Request timeout in seconds
        """
        try:
            self.client = Client(client_url, timeout=timeout)
            logger.info(f"Connected to FDSN client: {client_url}")
        except Exception as e:
            logger.error(f"Failed to initialize FDSN client: {e}")
            raise
    
    def get_stations(
        self,
        network: str = "*",
        station: str = "*",
        location: str = "*",
        channel: str = "BHZ",  # Broadband High-gain Vertical
        minlatitude: float = -90.0,
        maxlatitude: float = 90.0,
        minlongitude: float = -180.0,
        maxlongitude: float = 180.0,
        starttime: Optional[UTCDateTime] = None,
        endtime: Optional[UTCDateTime] = None
    ) -> List[dict]:
        """
        Get available seismic stations matching criteria.
        
        Args:
            network: Network code (e.g., "IU" for Global Seismographic Network)
            station: Station code
            location: Location code
            channel: Channel code (BHZ, HHZ, etc.)
            minlatitude, maxlatitude: Latitude bounds
            minlongitude, maxlongitude: Longitude bounds
            starttime, endtime: Time range for availability check
            
        Returns:
            List of station dictionaries
        """
        try:
            inventory = self.client.get_stations(
                network=network,
                station=station,
                location=location,
                channel=channel,
                minlatitude=minlatitude,
                maxlatitude=maxlatitude,
                minlongitude=minlongitude,
                maxlongitude=maxlongitude,
                starttime=starttime,
                endtime=endtime,
                level="station"
            )
            
            stations = []
            for net in inventory:
                for sta in net:
                    stations.append({
                        'network': net.code,
                        'station': sta.code,
                        'latitude': sta.latitude,
                        'longitude': sta.longitude,
                        'elevation': sta.elevation
                    })
            
            logger.info(f"Found {len(stations)} stations")
            return stations
            
        except Exception as e:
            logger.error(f"Failed to fetch stations: {e}")
            return []
    
    def fetch_waveform(
        self,
        network: str,
        station: str,
        starttime: UTCDateTime,
        endtime: UTCDateTime,
        location: str = "*",
        channel: str = "BHZ",
        attach_response: bool = False
    ) -> Optional[Stream]:
        """
        Fetch seismic waveform data for a specific station and time range.
        
        Args:
            network: Network code
            station: Station code
            location: Location code
            channel: Channel code
            starttime: Start time (UTCDateTime)
            endtime: End time (UTCDateTime)
            attach_response: Whether to attach instrument response metadata
            
        Returns:
            ObsPy Stream object or None if fetch fails
        """
        try:
            stream = self.client.get_waveforms(
                network=network,
                station=station,
                location=location,
                channel=channel,
                starttime=starttime,
                endtime=endtime,
                attach_response=attach_response
            )
            
            # Preprocess: detrend and filter
            stream.detrend('linear')
            stream.filter('bandpass', freqmin=0.1, freqmax=10.0)
            
            logger.info(
                f"Fetched {len(stream)} traces from {network}.{station} "
                f"({starttime} to {endtime})"
            )
            
            return stream
            
        except Exception as e:
            logger.warning(
                f"Failed to fetch waveform from {network}.{station}: {e}"
            )
            return None
    
    def fetch_recent_data(
        self,
        network: str,
        station: str,
        duration_minutes: int = 10,
        location: str = "*",
        channel: str = "BHZ"
    ) -> Optional[Stream]:
        """
        Fetch recent seismic data (convenience method).
        
        Args:
            network: Network code
            station: Station code
            duration_minutes: How many minutes of data to fetch
            location: Location code
            channel: Channel code
            
        Returns:
            ObsPy Stream object or None
        """
        endtime = UTCDateTime.now()
        starttime = endtime - (duration_minutes * 60)
        
        return self.fetch_waveform(
            network=network,
            station=station,
            location=location,
            channel=channel,
            starttime=starttime,
            endtime=endtime
        )
    
    def fetch_multiple_stations(
        self,
        stations: List[Tuple[str, str]],  # List of (network, station) tuples
        starttime: UTCDateTime,
        endtime: UTCDateTime,
        channel: str = "BHZ"
    ) -> Stream:
        """
        Fetch waveforms from multiple stations.
        
        Args:
            stations: List of (network, station) tuples
            starttime: Start time
            endtime: End time
            channel: Channel code
            
        Returns:
            Combined ObsPy Stream
        """
        combined_stream = Stream()
        
        for network, station in stations:
            stream = self.fetch_waveform(
                network=network,
                station=station,
                channel=channel,
                starttime=starttime,
                endtime=endtime
            )
            
            if stream:
                combined_stream += stream
        
        return combined_stream

