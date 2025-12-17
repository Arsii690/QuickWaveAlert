"""
QuakeAlert Dashboard - Real-time Seismic Wave Analysis
Streamlit interface for monitoring live seismic data and P-wave detection.
"""

import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import time
import sys
from pathlib import Path

# Add app directory to path for imports
sys.path.append(str(Path(__file__).parent))

# Import ML utilities
try:
    from app.ml_utils import (
        load_models, load_metrics, load_training_data_sample,
        detect_data_drift, get_shap_explanation, get_lime_explanation,
        get_feature_explanations, SHAP_AVAILABLE, LIME_AVAILABLE
    )
except ImportError:
    # Fallback if ml_utils not available
    SHAP_AVAILABLE = False
    LIME_AVAILABLE = False
    def load_models(): return None, None, None, None
    def load_metrics(): return {}
    def load_training_data_sample(): return None
    def detect_data_drift(*args, **kwargs): return {}
    def get_shap_explanation(*args, **kwargs): return None
    def get_lime_explanation(*args, **kwargs): return None
    def get_feature_explanations(): return {}

# Page Config
st.set_page_config(
    page_title="QuakeAlert - Seismic Analysis",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern UI
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
    
    /* Global Styles */
    .stApp {
        background: linear-gradient(135deg, #0a0a0f 0%, #1a1a2e 50%, #16213e 100%);
        font-family: 'Outfit', sans-serif;
    }
    
    /* Main Header */
    .main-header {
        background: linear-gradient(90deg, #00d4ff 0%, #7c3aed 50%, #f72585 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 3rem;
        font-weight: 700;
        text-align: center;
        margin-bottom: 0.5rem;
        font-family: 'Outfit', sans-serif;
    }
    
    .sub-header {
        color: #94a3b8;
        text-align: center;
        font-size: 1.1rem;
        margin-bottom: 2rem;
        font-weight: 300;
    }
    
    /* Metric Cards */
    .metric-card {
        background: linear-gradient(145deg, rgba(30, 41, 59, 0.8) 0%, rgba(15, 23, 42, 0.9) 100%);
        border: 1px solid rgba(99, 102, 241, 0.3);
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
        backdrop-filter: blur(10px);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 40px rgba(99, 102, 241, 0.2);
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #00d4ff, #7c3aed);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-family: 'JetBrains Mono', monospace;
    }
    
    .metric-label {
        color: #94a3b8;
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-top: 0.5rem;
    }
    
    /* Status Badges */
    .status-online {
        background: linear-gradient(135deg, #10b981, #059669);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: 500;
        display: inline-block;
        animation: pulse 2s infinite;
    }
    
    .status-offline {
        background: linear-gradient(135deg, #ef4444, #dc2626);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: 500;
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.7; }
    }
    
    /* Alert Box */
    .alert-box {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.2) 0%, rgba(220, 38, 38, 0.3) 100%);
        border: 2px solid #ef4444;
        border-radius: 12px;
        padding: 1rem 1.5rem;
        margin: 1rem 0;
        animation: alertPulse 1s infinite;
    }
    
    @keyframes alertPulse {
        0%, 100% { border-color: #ef4444; }
        50% { border-color: #f87171; }
    }
    
    /* Info Cards */
    .info-card {
        background: linear-gradient(145deg, rgba(30, 41, 59, 0.6) 0%, rgba(15, 23, 42, 0.8) 100%);
        border: 1px solid rgba(148, 163, 184, 0.2);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 0.5rem 0;
    }
    
    /* Sidebar Styling */
    .css-1d391kg, [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
    }
    
    /* Button Styling */
    .stButton > button {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
        font-family: 'Outfit', sans-serif;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(99, 102, 241, 0.4);
    }
    
    /* Tab Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: rgba(30, 41, 59, 0.5);
        border-radius: 12px;
        padding: 8px;
        width: 100%;
        display: flex;
        justify-content: space-between;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        border-radius: 8px;
        color: #94a3b8;
        font-weight: 500;
        flex: 1;
        text-align: center;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #6366f1, #8b5cf6);
        color: white;
    }
    
    /* DataFrames */
    .stDataFrame {
        border-radius: 12px;
        overflow: hidden;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Live indicator */
    .live-indicator {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background: rgba(16, 185, 129, 0.2);
        border: 1px solid #10b981;
        border-radius: 20px;
        padding: 4px 12px;
        font-size: 0.85rem;
        color: #10b981;
    }
    
    .live-dot {
        width: 8px;
        height: 8px;
        background: #10b981;
        border-radius: 50%;
        animation: blink 1s infinite;
    }
    
    @keyframes blink {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.3; }
    }
</style>
""", unsafe_allow_html=True)

# Header with gradient text
st.markdown('<h1 class="main-header">🌊 QuakeAlert</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Real-time Seismic Wave Analysis & P-Wave Detection</p>', unsafe_allow_html=True)

# Detect environment and set appropriate API URL
import os
DEFAULT_API_URL = os.environ.get("API_URL", "http://127.0.0.1:8000")

# For Hugging Face Spaces, use localhost since both run in same container
if "SPACE_ID" in os.environ:
    DEFAULT_API_URL = "http://localhost:8000"

# API Configuration in sidebar
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    
    API_URL = st.text_input(
        "🔗 API URL",
        value=DEFAULT_API_URL,
    help="FastAPI backend URL"
)

    # Connection status
    try:
        response = requests.get(f"{API_URL}/", timeout=2)
        if response.status_code == 200:
            st.markdown('<div class="live-indicator"><span class="live-dot"></span> Connected</div>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="status-offline">⚫ Disconnected</span>', unsafe_allow_html=True)
    except:
        st.markdown('<span class="status-offline">⚫ Offline</span>', unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### 📡 Station Selection")
    
    # Popular stations with flags
popular_stations = {
        "🇺🇸 Albuquerque, NM (IU.ANMO)": {"network": "IU", "station": "ANMO"},
        "🇺🇸 College, Alaska (IU.COLA)": {"network": "IU", "station": "COLA"},
        "🇺🇸 Tucson, Arizona (US.TAU)": {"network": "US", "station": "TAU"},
        "🇯🇵 Matsushiro, Japan (IU.MAJO)": {"network": "IU", "station": "MAJO"},
        "🇩🇪 Black Forest, Germany (II.BFO)": {"network": "II", "station": "BFO"},
        "🇦🇺 Canberra, Australia (II.CAN)": {"network": "II", "station": "CAN"},
        "🔧 Custom Station": {"network": "", "station": ""}
    }
    
    selected_station = st.selectbox(
        "Choose Station",
        options=list(popular_stations.keys()),
        index=0
    )
    
    if "Custom" in selected_station:
        col1, col2 = st.columns(2)
        with col1:
            network = st.text_input("Network", value="IU", max_chars=2)
        with col2:
            station = st.text_input("Station", value="ANMO", max_chars=5)
else:
    network = popular_stations[selected_station]["network"]
    station = popular_stations[selected_station]["station"]
        st.markdown(f"**Network:** `{network}` | **Station:** `{station}`")
    
    st.markdown("---")
    st.markdown("### 🎛️ Parameters")
    
    location = st.text_input("Location Code", value="*", help="Use * for any location")
    st.caption("📍 **Location Code**: Specifies the location identifier for the seismic station. Use '*' to search all locations.")
    
    channel = st.selectbox("Channel", ["BHZ", "HHZ", "EHZ", "BH1", "BH2"], index=0, 
                          help="BHZ = Broadband High-gain Vertical")
    st.caption("📡 **Channel**: Seismic data channel type. BHZ = Broadband High-gain Vertical, HHZ = High-gain Vertical, EHZ = Extremely High-gain Vertical")
    
    duration = st.slider("Duration (minutes)", 1, 60, 10, help="Analysis time window")
    st.caption("⏱️ **Duration**: Time window in minutes for waveform analysis. Longer durations capture more events but take more time to process.")
    
    st.markdown("---")
    st.markdown("### 🔬 STA/LTA Settings")
    sta_window = st.slider("STA Window (sec)", 0.5, 5.0, 1.0)
    st.caption("📊 **STA Window**: Short-Term Average window in seconds. Measures recent signal activity. Typical: 1-5 seconds.")
    
    lta_window = st.slider("LTA Window (sec)", 10.0, 120.0, 60.0)
    st.caption("📈 **LTA Window**: Long-Term Average window in seconds. Measures background noise level. Typical: 30-120 seconds.")
    
    threshold = st.slider("Detection Threshold", 2.0, 15.0, 5.0)
    st.caption("🎯 **Detection Threshold**: STA/LTA ratio threshold for event detection. Higher = fewer but more significant events. Typical: 3-10.")

# Main Content Area
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
    "📊 Real-time Analysis", 
    "🗺️ Station Map", 
    "📈 Statistics", 
    "📉 Data Drift",
    "🔍 Model Explainability",
    "🧠 SHAP/LIME",
    "📊 EDA",
    "ℹ️ About"
])

with tab1:
    # Top metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">--</div>
            <div class="metric-label">Events Today</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{network}.{station}</div>
            <div class="metric-label">Active Station</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{channel}</div>
            <div class="metric-label">Channel</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{duration}m</div>
            <div class="metric-label">Window</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Analysis Section
    col_main, col_side = st.columns([3, 1])
    
    with col_main:
        analyze_btn = st.button("🔍 Analyze Waveform", type="primary", use_container_width=True)
        
        if analyze_btn:
            with st.spinner(f"📡 Fetching data from {network}.{station}..."):
                try:
                    payload = {
                        "network": network,
                        "station": station,
                        "location": location,
                        "channel": channel,
                        "duration_minutes": duration
                    }
                    
                    response = requests.post(
                        f"{API_URL}/analyze_waveform",
                        json=payload,
                        timeout=120
                    )
                    
                    # Check for HTTP errors
                    if response.status_code == 404:
                        error_data = response.json()
                        error_detail = error_data.get("detail", "No data available")
                        st.error(f"❌ No Data Found: {error_detail}")
                        st.info("💡 **Tip**: Try a different channel (BHZ, HHZ, EHZ) or check if the station has data available for the selected channel and time period.")
                        st.stop()
                    elif response.status_code != 200:
                        st.error(f"❌ Server Error: {response.status_code}")
                        try:
                            error_data = response.json()
                            st.error(f"Details: {error_data.get('detail', 'Unknown error')}")
                        except:
                            st.error(f"Response: {response.text[:200]}")
                        st.stop()
                    
                    if response.status_code == 200:
                        result = response.json()
                        
                        # Success message with animation
                        st.success("✅ Analysis Complete!")
                        
                        analysis = result.get("analysis", {})
                        events = analysis.get("events", [])
                        num_events = analysis.get("events_detected", 0)
                        
                        # Update metrics dynamically
                        st.markdown(f"""
                        <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 1rem; margin: 1rem 0;">
                            <div class="metric-card">
                                <div class="metric-value">{num_events}</div>
                                <div class="metric-label">Events Detected</div>
                            </div>
                            <div class="metric-card">
                                <div class="metric-value">{network}.{station}</div>
                                <div class="metric-label">Station</div>
                            </div>
                            <div class="metric-card">
                                <div class="metric-value">{channel}</div>
                                <div class="metric-label">Channel</div>
                            </div>
                            <div class="metric-card">
                                <div class="metric-value">{duration}m</div>
                                <div class="metric-label">Duration</div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Events Display
                        if events:
                            st.markdown("### 🎯 Detected Seismic Events")
                            st.caption("📊 **Detected Events**: Seismic events identified by the STA/LTA algorithm. Each event includes timing, amplitude, and frequency characteristics.")
                            
                            events_df = pd.DataFrame(events)
                            
                            # Store in session state for drift detection
                            st.session_state.recent_events_df = events_df
                            
                            # Check for significant events
                            if 'peak_ratio' in events_df.columns:
                                high_ratio = events_df[events_df['peak_ratio'] > 10.0]
                                if len(high_ratio) > 0:
                                    st.markdown(f"""
                                    <div class="alert-box">
                                        <h3 style="margin: 0; color: #ef4444;">
                                            ⚠️ {len(high_ratio)} Significant Event(s) Detected!
                                        </h3>
                                        <p style="margin: 0.5rem 0 0 0; color: #f87171;">
                                            STA/LTA ratio exceeded 10.0 - Potential earthquake activity
                                        </p>
                                    </div>
                                    """, unsafe_allow_html=True)
                            
                            # Events Table
                            display_cols = ['start_time', 'peak_time', 'peak_ratio', 'peak_amplitude', 'duration', 'dominant_frequency']
                            available_cols = [col for col in display_cols if col in events_df.columns]
                            
                            if available_cols:
                                st.dataframe(
                                    events_df[available_cols].style.format({
                                        'peak_ratio': '{:.2f}',
                                        'peak_amplitude': '{:.2e}',
                                        'duration': '{:.2f}s',
                                        'dominant_frequency': '{:.2f} Hz'
                                    }).background_gradient(subset=['peak_ratio'], cmap='Reds'),
                                    use_container_width=True,
                                    height=250
                                )
                            
                            # Visualizations
                            st.markdown("### 📊 Event Visualizations")
                            
                            viz_col1, viz_col2 = st.columns(2)
                            
                            with viz_col1:
                            if 'peak_ratio' in events_df.columns:
                                fig_ratio = go.Figure()
                                    
                                    # Color based on severity
                                    colors = ['#10b981' if x <= 5 else '#f59e0b' if x <= 10 else '#ef4444' 
                                             for x in events_df['peak_ratio']]
                                    
                                fig_ratio.add_trace(go.Bar(
                                        x=list(range(len(events_df))),
                                    y=events_df['peak_ratio'],
                                        marker_color=colors,
                                        text=events_df['peak_ratio'].round(2),
                                        textposition='auto'
                                    ))
                                    
                                    # Add threshold line
                                    fig_ratio.add_hline(y=threshold, line_dash="dash", 
                                                       line_color="#f59e0b", 
                                                       annotation_text="Threshold")
                                    
                                fig_ratio.update_layout(
                                        title="STA/LTA Ratio by Event",
                                        xaxis_title="Event #",
                                    yaxis_title="STA/LTA Ratio",
                                        template="plotly_dark",
                                        paper_bgcolor='rgba(0,0,0,0)',
                                        plot_bgcolor='rgba(0,0,0,0)',
                                        height=350
                                )
                                st.plotly_chart(fig_ratio, use_container_width=True)
                            
                            with viz_col2:
                            if 'peak_amplitude' in events_df.columns:
                                fig_amp = go.Figure()
                                    
                                fig_amp.add_trace(go.Scatter(
                                        x=list(range(len(events_df))),
                                    y=events_df['peak_amplitude'],
                                    mode='markers+lines',
                                        marker=dict(
                                            size=12,
                                            color=events_df.get('peak_ratio', [5]*len(events_df)),
                                            colorscale='Turbo',
                                            showscale=True,
                                            colorbar=dict(title="Ratio")
                                        ),
                                        line=dict(color='#6366f1', width=2)
                                    ))
                                    
                                fig_amp.update_layout(
                                        title="Peak Amplitude by Event",
                                        xaxis_title="Event #",
                                    yaxis_title="Amplitude",
                                        yaxis_type="log",
                                        template="plotly_dark",
                                        paper_bgcolor='rgba(0,0,0,0)',
                                        plot_bgcolor='rgba(0,0,0,0)',
                                        height=350
                                )
                                st.plotly_chart(fig_amp, use_container_width=True)
                            
                            # Frequency Analysis
                            if 'dominant_frequency' in events_df.columns:
                                fig_freq = go.Figure()
                                
                                fig_freq.add_trace(go.Histogram(
                                    x=events_df['dominant_frequency'],
                                    nbinsx=20,
                                    marker_color='#8b5cf6',
                                    opacity=0.8
                                ))
                                
                                fig_freq.update_layout(
                                    title="Dominant Frequency Distribution",
                                    xaxis_title="Frequency (Hz)",
                                    yaxis_title="Count",
                                    template="plotly_dark",
                                    paper_bgcolor='rgba(0,0,0,0)',
                                    plot_bgcolor='rgba(0,0,0,0)',
                                    height=300
                                )
                                st.plotly_chart(fig_freq, use_container_width=True)
                        
                        else:
                            st.info("✨ No seismic events detected in this time window - All quiet!")
                            st.balloons()
                    
                    else:
                        st.error(f"❌ API Error: {response.status_code}")
                        st.json(response.json())
                        
                except requests.exceptions.ConnectionError:
                    st.error("🔌 Connection Failed!")
                    st.markdown("""
                    <div class="info-card">
                        <h4>Start the FastAPI server:</h4>
                        <code style="background: #1e293b; padding: 0.5rem 1rem; border-radius: 8px; display: block; margin-top: 0.5rem;">
                            uvicorn app.main:app --reload --port 8000
                        </code>
                    </div>
                    """, unsafe_allow_html=True)
                except requests.exceptions.Timeout:
                    st.warning("⏱️ Request timed out. Try reducing the duration or check server status.")
                except Exception as e:
                    error_msg = str(e)
                    if "No data" in error_msg or "404" in error_msg or "not found" in error_msg.lower():
                        st.error(f"❌ No Data Found: {error_msg}")
                        st.info("💡 **Tip**: Try a different channel (BHZ, HHZ, EHZ) or check if the station has data available for the selected channel and time period.")
                    else:
                    st.error(f"❌ Error: {e}")
    
    with col_side:
        st.markdown("### ⚡ Quick Actions")
        
        if st.button("🔄 Refresh", use_container_width=True):
            st.rerun()
        
        if st.button("🏥 Health Check", use_container_width=True):
            try:
                response = requests.get(f"{API_URL}/", timeout=5)
                if response.status_code == 200:
                    st.success("✅ API Online")
                    data = response.json()
                    st.json(data)
                else:
                    st.error("❌ API Error")
            except:
                st.error("❌ Cannot reach API")
        
        st.markdown("---")
        st.markdown("### 📋 Current Config")
        st.markdown(f"""
        - **Network:** `{network}`
        - **Station:** `{station}`
        - **Channel:** `{channel}`
        - **Duration:** `{duration} min`
        - **STA:** `{sta_window}s`
        - **LTA:** `{lta_window}s`
        - **Threshold:** `{threshold}`
        """)

with tab2:
    st.markdown("### 🗺️ Global Seismic Stations")
    st.caption("🌍 **Station Map**: Interactive map showing seismic monitoring stations worldwide. Each station collects real-time seismic data.")
    
    # Sample station locations for visualization
    station_data = {
        "name": ["Albuquerque (ANMO)", "College (COLA)", "Tucson (TAU)", "Matsushiro (MAJO)", 
                 "Black Forest (BFO)", "Canberra (CAN)"],
        "lat": [34.95, 64.87, 32.22, 36.54, 48.33, -35.32],
        "lon": [-106.46, -147.86, -110.93, 138.21, 8.33, 149.00],
        "network": ["IU", "IU", "US", "IU", "II", "II"]
    }
    
    stations_df = pd.DataFrame(station_data)
    
    fig_map = px.scatter_geo(
        stations_df,
        lat="lat",
        lon="lon",
        hover_name="name",
        color="network",
        projection="natural earth",
        title="Seismic Station Network",
        color_discrete_map={"IU": "#6366f1", "US": "#10b981", "II": "#f59e0b"}
    )
    
    fig_map.update_layout(
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        geo=dict(
            bgcolor='rgba(0,0,0,0)',
            landcolor='#1e293b',
            oceancolor='#0f172a',
            showocean=True,
            showland=True
        ),
        height=500
    )
    
    st.plotly_chart(fig_map, use_container_width=True)
    
    # Station search
    st.markdown("### 🔍 Search Stations")
    
    if st.button("Search Available Stations"):
        with st.spinner("Fetching station information..."):
            try:
                response = requests.post(
                    f"{API_URL}/stations",
                    json={"network": "*", "station": "*"},
                    timeout=30
                )
                
                if response.status_code == 200:
                    result = response.json()
                    stations = result.get("stations", [])
                    st.success(f"Found {len(stations)} stations")
                    
                    if stations:
                        st.dataframe(pd.DataFrame(stations), use_container_width=True)
                else:
                    st.error(f"Error: {response.status_code}")
            except Exception as e:
                st.error(f"Error: {e}")

with tab3:
    st.markdown("### 📈 Analysis Statistics")
    st.caption("📊 **Statistics**: Performance metrics and analysis statistics for detection algorithms and ML models.")
    
    col1, col2 = st.columns(2)
    
    with col1:
    st.markdown("""
        <div class="info-card">
            <h4>🎯 Detection Performance</h4>
            <p>STA/LTA algorithm parameters:</p>
            <ul>
                <li><strong>STA Window:</strong> Short-term average window (1-5 sec)</li>
                <li><strong>LTA Window:</strong> Long-term average window (30-120 sec)</li>
                <li><strong>Threshold:</strong> Detection trigger level (2-15)</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="info-card">
            <h4>📊 Feature Extraction</h4>
            <p>For each detected event, we extract:</p>
            <ul>
                <li>Peak amplitude & RMS amplitude</li>
                <li>Dominant frequency & spectral centroid</li>
                <li>Event duration & timing</li>
                <li>Signal-to-noise ratio</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Model Evaluation Metrics
    st.markdown("### 🤖 Model Evaluation Metrics")
    st.caption("📊 **Model Metrics**: Performance metrics for all trained ML models. Higher values indicate better performance.")
    
    # Load metrics
    metrics = load_metrics()
    
    # Display metrics in a table
    metrics_df = pd.DataFrame({
        'Model': ['Random Forest', 'Gradient Boosting', 'Logistic Regression'],
        'Accuracy': [
            metrics.get('random_forest', {}).get('accuracy', 0),
            metrics.get('gradient_boosting', {}).get('accuracy', 0),
            metrics.get('logistic_regression', {}).get('accuracy', 0)
        ],
        'Precision': [
            metrics.get('random_forest', {}).get('precision', 0),
            metrics.get('gradient_boosting', {}).get('precision', 0),
            metrics.get('logistic_regression', {}).get('precision', 0)
        ],
        'Recall': [
            metrics.get('random_forest', {}).get('recall', 0),
            metrics.get('gradient_boosting', {}).get('recall', 0),
            metrics.get('logistic_regression', {}).get('recall', 0)
        ],
        'F1-Score': [
            metrics.get('random_forest', {}).get('f1_score', 0),
            metrics.get('gradient_boosting', {}).get('f1_score', 0),
            metrics.get('logistic_regression', {}).get('f1_score', 0)
        ],
        'ROC-AUC': [
            metrics.get('random_forest', {}).get('roc_auc', 0),
            metrics.get('gradient_boosting', {}).get('roc_auc', 0),
            metrics.get('logistic_regression', {}).get('roc_auc', 0)
        ]
    })
    
    st.dataframe(metrics_df.style.format({
        'Accuracy': '{:.4f}',
        'Precision': '{:.4f}',
        'Recall': '{:.4f}',
        'F1-Score': '{:.4f}',
        'ROC-AUC': '{:.4f}'
    }).background_gradient(subset=['Accuracy', 'F1-Score', 'ROC-AUC']), use_container_width=True)
    
    # Model comparison charts
    st.markdown("### 📊 Model Comparison Charts")
    
    chart_col1, chart_col2 = st.columns(2)
    
    with chart_col1:
        # Accuracy comparison
        fig_acc = go.Figure()
        fig_acc.add_trace(go.Bar(
            x=metrics_df['Model'],
            y=metrics_df['Accuracy'],
            marker_color=['#6366f1', '#8b5cf6', '#ec4899'],
            text=metrics_df['Accuracy'].round(4),
            textposition='auto'
        ))
        fig_acc.update_layout(
            title="Accuracy Comparison",
            yaxis_title="Accuracy",
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            height=350
        )
        st.plotly_chart(fig_acc, use_container_width=True)
    
    with chart_col2:
        # F1-Score comparison
        fig_f1 = go.Figure()
        fig_f1.add_trace(go.Bar(
            x=metrics_df['Model'],
            y=metrics_df['F1-Score'],
            marker_color=['#10b981', '#059669', '#047857'],
            text=metrics_df['F1-Score'].round(4),
            textposition='auto'
        ))
        fig_f1.update_layout(
            title="F1-Score Comparison",
            yaxis_title="F1-Score",
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            height=350
        )
        st.plotly_chart(fig_f1, use_container_width=True)
    
    # ROC-AUC comparison
    fig_roc = go.Figure()
    fig_roc.add_trace(go.Bar(
        x=metrics_df['Model'],
        y=metrics_df['ROC-AUC'],
        marker_color=['#f59e0b', '#d97706', '#b45309'],
        text=metrics_df['ROC-AUC'].round(4),
        textposition='auto'
    ))
    fig_roc.update_layout(
        title="ROC-AUC Comparison",
        yaxis_title="ROC-AUC",
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=350
    )
    st.plotly_chart(fig_roc, use_container_width=True)
    
    # All metrics radar chart
    st.markdown("### 🎯 Comprehensive Model Performance")
    fig_radar = go.Figure()
    
    for idx, model in enumerate(metrics_df['Model']):
        fig_radar.add_trace(go.Scatterpolar(
            r=[
                metrics_df.iloc[idx]['Accuracy'],
                metrics_df.iloc[idx]['Precision'],
                metrics_df.iloc[idx]['Recall'],
                metrics_df.iloc[idx]['F1-Score'],
                metrics_df.iloc[idx]['ROC-AUC']
            ],
            theta=['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC'],
            fill='toself',
            name=model
        ))
    
    fig_radar.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )),
        showlegend=True,
        title="Model Performance Radar Chart",
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        height=500
    )
    st.plotly_chart(fig_radar, use_container_width=True)

# Data Drift Detection Tab
with tab4:
    st.markdown("### 📉 Data Drift Detection")
    st.caption("🔍 **Data Drift**: Monitors if production data distribution differs from training data. Drift can degrade model performance.")
    
    st.info("💡 **How it works**: Compares current production data statistics with training data baseline using Kolmogorov-Smirnov tests and statistical measures.")
    
    # Load training data
    training_data = load_training_data_sample()
    
    if training_data is not None:
        # Get production data from recent analysis
        if 'recent_events_df' in st.session_state and not st.session_state.recent_events_df.empty:
            production_data = st.session_state.recent_events_df
            
            # Detect drift
            drift_results = detect_data_drift(production_data, training_data)
            
            if drift_results:
                st.markdown("#### 🎯 Drift Detection Results")
                
                # Summary
                drift_scores = [v['drift_score'] for v in drift_results.values()]
                avg_drift = np.mean(drift_scores)
                max_drift = max(drift_scores)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Average Drift Score", f"{avg_drift:.3f}", 
                             delta="High" if avg_drift > 0.3 else "Low" if avg_drift < 0.2 else "Medium")
                with col2:
                    st.metric("Max Drift Score", f"{max_drift:.3f}")
                with col3:
                    features_with_drift = sum(1 for v in drift_results.values() if v['has_drift'])
                    st.metric("Features with Drift", f"{features_with_drift}/{len(drift_results)}")
                
                # Drift details table - convert has_drift to text for better visibility
                drift_df = pd.DataFrame({
                    'Feature': list(drift_results.keys()),
                    'Drift Score': [v['drift_score'] for v in drift_results.values()],
                    'P-Value': [v['p_value'] for v in drift_results.values()],
                    'Has Drift': ['✅ Yes' if v['has_drift'] else '❌ No' for v in drift_results.values()]
                })
                
                st.dataframe(drift_df.style.format({
                    'Drift Score': '{:.4f}',
                    'P-Value': '{:.4f}'
                }).background_gradient(subset=['Drift Score']), use_container_width=True)
                
                # Distribution comparison charts
                st.markdown("#### 📊 Distribution Comparison")
                
                for feature in list(drift_results.keys())[:3]:  # Show first 3 features
                    if feature in production_data.columns and feature in training_data.columns:
                        fig = go.Figure()
                        
                        # Production distribution
                        fig.add_trace(go.Histogram(
                            x=production_data[feature].dropna(),
                            name='Production',
                            opacity=0.7,
                            marker_color='#ef4444'
                        ))
                        
                        # Training distribution
                        fig.add_trace(go.Histogram(
                            x=training_data[feature].dropna(),
                            name='Training',
                            opacity=0.7,
                            marker_color='#8b5cf6'
                        ))
                        
                        fig.update_layout(
                            title=f"{feature} Distribution Comparison",
                            xaxis_title=feature,
                            yaxis_title="Frequency",
                            barmode='overlay',
                            template="plotly_dark",
                            paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(0,0,0,0)',
                            height=350
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("⚠️ No drift data available. Run an analysis first to compare with training data.")
        else:
            st.info("ℹ️ Run a waveform analysis first to detect data drift. Production data will be compared with training baseline.")
    else:
        st.warning("⚠️ Training data not available. Cannot perform drift detection.")

# Model Explainability Tab
with tab5:
    st.markdown("### 🔍 Model Explainability")
    st.caption("🧠 **Explainability**: Understand how each feature contributes to model predictions and get detailed feature explanations.")
    
    # Load models
    rf_model, gb_model, lr_model, scaler = load_models()
    feature_explanations = get_feature_explanations()
    
    if rf_model is None and gb_model is None and lr_model is None:
        st.warning("⚠️ No trained models found. Run the training pipeline first.")
    else:
        # Feature selection
        st.markdown("#### 📋 Feature Explanations")
        selected_feature = st.selectbox(
            "Select Feature to Explain",
            options=list(feature_explanations.keys()),
            help="Choose a feature to see its explanation and impact on predictions"
        )
        
        if selected_feature:
            st.markdown(f"#### {selected_feature.replace('_', ' ').title()}")
            explanation = feature_explanations.get(selected_feature, "No explanation available.")
            st.markdown(explanation)
        
        # Model-specific explanations
        st.markdown("#### 🤖 Model Predictions Explanation")
        
        # Get sample data for explanation
        if 'recent_events_df' in st.session_state and not st.session_state.recent_events_df.empty:
            sample_data = st.session_state.recent_events_df
            
            model_choice = st.selectbox("Select Model for Explanation", 
                                       ["Random Forest", "Gradient Boosting", "Logistic Regression"])
            
            if model_choice == "Random Forest" and rf_model:
                model = rf_model
            elif model_choice == "Gradient Boosting" and gb_model:
                model = gb_model
            elif model_choice == "Logistic Regression" and lr_model:
                model = lr_model
            else:
                model = None
            
            if model and scaler:
                # Import feature computation function
                from app.ml_utils import FEATURE_COLS, BASE_FEATURE_COLS, compute_derived_features, get_scaler_feature_order
                
                # Check if base features are available
                available_base = [col for col in BASE_FEATURE_COLS if col in sample_data.columns]
                
                if len(available_base) == len(BASE_FEATURE_COLS):
                    # Compute derived features to match training pipeline
                    sample_features = sample_data[BASE_FEATURE_COLS].iloc[:1].copy()
                    sample_features = compute_derived_features(sample_features)
                    
                    # Get the exact feature order that the scaler expects
                    # This is CRITICAL - must match the order used during training
                    expected_feature_order = get_scaler_feature_order(scaler)
                    
                    # Ensure all features exist (add missing ones with 0.0)
                    missing = [f for f in expected_feature_order if f not in sample_features.columns]
                    if missing:
                        for feat in missing:
                            sample_features[feat] = 0.0
                    
                    # CRITICAL: Select features in EXACT order that scaler expects
                    # pandas DataFrame selection with a list reorders columns to match the list
                    sample_features_ordered = sample_features[expected_feature_order]
                    
                    # Transform - scaler validates feature names and order match exactly
                    try:
                        scaled_features = scaler.transform(sample_features_ordered)
                    except ValueError as e:
                        # Provide detailed error info for debugging
                        st.error(f"❌ Feature order error: {str(e)[:200]}")
                        if hasattr(scaler, 'feature_names_in_'):
                            st.info(f"💡 Scaler expects: {list(scaler.feature_names_in_)}")
                        st.info(f"💡 We provided: {list(sample_features_ordered.columns)}")
                        scaled_features = None
                    
                    if scaled_features is not None:
                        # Get prediction
                        prediction = model.predict(scaled_features)[0]
                        probability = model.predict_proba(scaled_features)[0]
                    
                        st.markdown(f"**Prediction:** {'Significant Event' if prediction == 1 else 'Normal Event'}")
                        st.markdown(f"**Confidence:** {probability[prediction]:.2%}")
                    else:
                        st.error("❌ Could not make prediction due to feature order mismatch")
                    
        else:
            st.info("ℹ️ Run a waveform analysis first to see model explanations on real data.")

# SHAP/LIME Tab
with tab6:
    st.markdown("### 🧠 SHAP & LIME Explanations")
    st.caption("🔬 **SHAP/LIME**: Advanced explainability techniques that show how each feature contributes to individual predictions.")
    
    col_info1, col_info2 = st.columns(2)
    with col_info1:
        st.markdown("""
        <div class="info-card">
            <h4>📊 SHAP (SHapley Additive exPlanations)</h4>
            <p>SHAP values explain the output of any machine learning model using game theory. 
            Each feature gets a SHAP value showing its contribution to the prediction.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col_info2:
        st.markdown("""
        <div class="info-card">
            <h4>🍋 LIME (Local Interpretable Model-agnostic Explanations)</h4>
            <p>LIME explains individual predictions by approximating the model locally with an interpretable model. 
            It shows which features are most important for a specific prediction.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Check availability
    availability_col1, availability_col2 = st.columns(2)
    with availability_col1:
        if SHAP_AVAILABLE:
            st.success("✅ SHAP is available")
        else:
            st.error("❌ SHAP not installed. Run: `pip install shap`")
    
    with availability_col2:
        if LIME_AVAILABLE:
            st.success("✅ LIME is available")
        else:
            st.error("❌ LIME not installed. Run: `pip install lime`")
    
    # Generate explanations - use training data if recent_events_df not available
    rf_model, gb_model, lr_model, scaler = load_models()
    
    # Try to get data from recent events or training data
    sample_data = None
    if 'recent_events_df' in st.session_state:
        sample_data = st.session_state.recent_events_df
    else:
        # Use training data sample as fallback
        training_data = load_training_data_sample()
        if training_data is not None and not training_data.empty:
            sample_data = training_data
            st.info("ℹ️ Using training data sample for explanations. Run waveform analysis to use live data.")
    
    if (SHAP_AVAILABLE or LIME_AVAILABLE) and sample_data is not None:
        from app.ml_utils import FEATURE_COLS, BASE_FEATURE_COLS, compute_derived_features
        
        available_base = [col for col in BASE_FEATURE_COLS if col in sample_data.columns]
        
        if len(available_base) == len(BASE_FEATURE_COLS) and (rf_model or gb_model or lr_model):
            model_choice = st.selectbox("Select Model", 
                                       ["Random Forest", "Gradient Boosting", "Logistic Regression"])
            
            if model_choice == "Random Forest":
                model = rf_model
            elif model_choice == "Gradient Boosting":
                model = gb_model
            else:
                model = lr_model
            
            if model and scaler:
                # Compute derived features to match training
                sample_features = sample_data[BASE_FEATURE_COLS].iloc[:100].copy()
                sample_features = compute_derived_features(sample_features)
                
                # Get expected feature order from scaler
                from app.ml_utils import get_scaler_feature_order
                expected_feature_order = get_scaler_feature_order(scaler)
                
                # Ensure all features are present
                for feat in expected_feature_order:
                    if feat not in sample_features.columns:
                        sample_features[feat] = 0.0
                
                # Select in exact order
                sample_features_ordered = sample_features[expected_feature_order]
                
                # Transform using DataFrame (scaler validates feature names and order)
                try:
                    scaled_features = scaler.transform(sample_features_ordered)
                except ValueError as e:
                    st.error(f"❌ Feature order error: {str(e)[:200]}")
                    scaled_features = None
                
                if SHAP_AVAILABLE:
                    st.markdown("#### 📊 SHAP Explanation")
                    if st.button("Generate SHAP Explanation"):
                        with st.spinner("Computing SHAP values (this may take a moment)..."):
                            # Use scaled features for SHAP
                            shap_values = get_shap_explanation(model, pd.DataFrame(scaled_features, columns=FEATURE_COLS), FEATURE_COLS)
                            if shap_values is not None:
                                st.success("✅ SHAP values computed!")
                                st.info("💡 SHAP values show how each feature contributes to the prediction. Positive values push toward 'Significant Event', negative toward 'Normal Event'.")
                                
                                # Handle SHAP values - for binary classification, it returns a list
                                if isinstance(shap_values, list):
                                    # For binary classification, use the positive class (index 1)
                                    if len(shap_values) == 2:
                                        shap_values = shap_values[1]
                                    else:
                                        shap_values = shap_values[0]
                                
                                # Ensure shap_values is 2D
                                if len(shap_values.shape) == 1:
                                    shap_values = shap_values.reshape(1, -1)
                                
                                shap_df = pd.DataFrame(
                                    shap_values,
                                    columns=FEATURE_COLS
                                )
                                
                                # Mean absolute SHAP values
                                mean_shap = shap_df.abs().mean().sort_values(ascending=False)
                                
                                fig_shap = go.Figure()
                                fig_shap.add_trace(go.Bar(
                                    x=mean_shap.values,
                                    y=mean_shap.index,
                                    orientation='h',
                                    marker_color='#8b5cf6'
                                ))
                                fig_shap.update_layout(
                                    title="Mean Absolute SHAP Values (Feature Importance)",
                                    xaxis_title="Mean |SHAP Value|",
                                    template="plotly_dark",
                                    paper_bgcolor='rgba(0,0,0,0)',
                                    plot_bgcolor='rgba(0,0,0,0)',
                                    height=400
                                )
                                st.plotly_chart(fig_shap, use_container_width=True)
                
                if LIME_AVAILABLE and scaled_features is not None:
                    st.markdown("#### 🍋 LIME Explanation")
                    max_idx = max(0, len(scaled_features) - 1)
                    instance_idx = st.slider("Select Instance", 0, max_idx, 0, key="lime_instance")
                    
                    if st.button("Generate LIME Explanation"):
                        with st.spinner("Computing LIME explanation..."):
                            # Use scaled features for LIME
                            scaled_df = pd.DataFrame(scaled_features, columns=FEATURE_COLS)
                            lime_exp = get_lime_explanation(model, scaled_df, FEATURE_COLS, instance_idx)
                            if lime_exp:
                                st.success("✅ LIME explanation generated!")
                                
                                # Display explanation
                                lime_df = pd.DataFrame({
                                    'Feature': lime_exp['features'],
                                    'Contribution': lime_exp['contributions']
                                }).sort_values('Contribution', key=abs, ascending=False)
                                
                                st.dataframe(lime_df, use_container_width=True)
                                
                                # Visualization
                                fig_lime = go.Figure()
                                colors = ['#10b981' if x > 0 else '#ef4444' for x in lime_df['Contribution']]
                                fig_lime.add_trace(go.Bar(
                                    x=lime_df['Contribution'],
                                    y=lime_df['Feature'],
                                    orientation='h',
                                    marker_color=colors
                                ))
                                fig_lime.update_layout(
                                    title=f"LIME Explanation for Instance {instance_idx}",
                                    xaxis_title="Contribution to Prediction",
                                    template="plotly_dark",
                                    paper_bgcolor='rgba(0,0,0,0)',
                                    plot_bgcolor='rgba(0,0,0,0)',
                                    height=400
                                )
                                st.plotly_chart(fig_lime, use_container_width=True)
                                
                                st.info(f"**Prediction Probability:** {lime_exp['prediction']:.2%}")
        else:
            st.warning("⚠️ Required features not found in data. Ensure data contains all base features.")
    elif not (SHAP_AVAILABLE or LIME_AVAILABLE):
        st.warning("⚠️ SHAP and LIME are not available. Install with: `pip install shap lime`")
    else:
        st.info("ℹ️ No data available. Run a waveform analysis or ensure training data is available.")

# EDA Tab
with tab7:
    st.markdown("### 📊 Exploratory Data Analysis (EDA)")
    st.caption("🔍 **EDA**: Comprehensive analysis of the dataset to understand patterns, distributions, and relationships.")
    
    # Load data
    training_data = load_training_data_sample()
    
    if training_data is not None and not training_data.empty:
        # Dataset overview
        st.markdown("#### 📋 Dataset Overview")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Samples", len(training_data))
        with col2:
            st.metric("Features", len(training_data.columns))
        with col3:
            st.metric("Missing Values", training_data.isnull().sum().sum())
        with col4:
            st.metric("Memory Usage", f"{training_data.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        # Statistical summary
        st.markdown("#### 📈 Statistical Summary")
        st.dataframe(training_data.describe(), use_container_width=True)
        
        # Feature distributions
        st.markdown("#### 📊 Feature Distributions")
        feature_cols = ['peak_ratio', 'peak_amplitude', 'rms_amplitude', 
                       'duration', 'dominant_frequency', 'spectral_centroid']
        available_features = [f for f in feature_cols if f in training_data.columns]
        
        selected_eda_feature = st.selectbox("Select Feature for Distribution", available_features)
        
        if selected_eda_feature:
            fig_dist = go.Figure()
            fig_dist.add_trace(go.Histogram(
                x=training_data[selected_eda_feature].dropna(),
                nbinsx=30,
                marker_color='#8b5cf6',
                opacity=0.7
            ))
            fig_dist.update_layout(
                title=f"{selected_eda_feature} Distribution",
                xaxis_title=selected_eda_feature,
                yaxis_title="Frequency",
                template="plotly_dark",
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                height=400
            )
            st.plotly_chart(fig_dist, use_container_width=True)
        
        # Correlation matrix
        st.markdown("#### 🔗 Feature Correlations")
        numeric_cols = training_data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            corr_matrix = training_data[numeric_cols].corr()
            
            fig_corr = go.Figure(data=go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.columns,
                colorscale='RdBu',
                zmid=0,
                text=corr_matrix.round(2).values,
                texttemplate="%{text}",
                textfont={"size": 10}
            ))
            fig_corr.update_layout(
                title="Correlation Matrix",
                template="plotly_dark",
                paper_bgcolor='rgba(0,0,0,0)',
                height=500
            )
            st.plotly_chart(fig_corr, use_container_width=True)
        
        # Box plots for outliers
        st.markdown("#### 📦 Outlier Detection (Box Plots)")
        if available_features:
            selected_box_feature = st.selectbox("Select Feature for Box Plot", available_features, key="box_plot")
            
            fig_box = go.Figure()
            fig_box.add_trace(go.Box(
                y=training_data[selected_box_feature].dropna(),
                name=selected_box_feature,
                marker_color='#8b5cf6'
            ))
            fig_box.update_layout(
                title=f"{selected_box_feature} Box Plot",
                yaxis_title=selected_box_feature,
                template="plotly_dark",
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                height=400
            )
            st.plotly_chart(fig_box, use_container_width=True)
        
        # Pair plots (sample)
        st.markdown("#### 🔄 Feature Relationships")
        if len(available_features) >= 2:
            x_feature = st.selectbox("X-axis Feature", available_features, key="pair_x")
            y_feature = st.selectbox("Y-axis Feature", available_features, key="pair_y")
            
            if x_feature != y_feature:
                fig_scatter = go.Figure()
                fig_scatter.add_trace(go.Scatter(
                    x=training_data[x_feature].dropna(),
                    y=training_data[y_feature].dropna(),
                    mode='markers',
                    marker=dict(
                        color='#6366f1',
                        size=5,
                        opacity=0.6
                    )
                ))
                fig_scatter.update_layout(
                    title=f"{x_feature} vs {y_feature}",
                    xaxis_title=x_feature,
                    yaxis_title=y_feature,
                    template="plotly_dark",
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    height=400
                )
                st.plotly_chart(fig_scatter, use_container_width=True)
    else:
        st.warning("⚠️ Training data not available for EDA. Run backfill.py to collect data.")

with tab8:
    st.markdown("### ℹ️ About QuakeAlert")
    
    st.markdown("""
    <div class="info-card">
        <h3>🌊 Real-time Seismic Wave Analysis</h3>
        <p style="font-size: 1.1rem; line-height: 1.8;">
            <strong>QuakeAlert</strong> is an advanced seismic monitoring system that uses machine learning 
            and signal processing to detect earthquakes in real-time from live seismic data streams.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        #### 🔬 How It Works
        
        1. **Data Acquisition**
           - Fetches live waveform data from FDSN
           - Supports multiple global seismic networks
        
        2. **Signal Processing**
           - STA/LTA algorithm for P-wave detection
           - Bandpass filtering for noise reduction
        
        3. **Feature Extraction**
           - Spectral analysis
           - Amplitude measurements
           - Timing characteristics
        
        4. **ML Classification**
           - Random Forest, Gradient Boosting, Logistic Regression
           - Trained on historical seismic events
        """)
    
    with col2:
        st.markdown("""
        #### 🛠️ Technology Stack
        
        | Component | Technology |
        |-----------|------------|
        | Backend | FastAPI + ObsPy |
        | ML Pipeline | Prefect |
        | Feature Store | Hopsworks |
        | Model Registry | Hopsworks |
        | Frontend | Streamlit |
        | Deployment | Docker + HF Spaces |
        | CI/CD | GitHub Actions |
        
        #### 📡 Data Sources
        
        - IRIS (Incorporated Research Institutions for Seismology)
        - FDSN (Federation of Digital Seismograph Networks)
        """)
    
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #64748b; padding: 2rem;">
        <p>Built with ❤️ for seismic research and earthquake preparedness</p>
        <p style="font-size: 0.9rem;">© 2025 QuakeAlert</p>
    </div>
    """, unsafe_allow_html=True)
