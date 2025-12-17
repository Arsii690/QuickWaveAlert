"""
ML Utilities for Dashboard
Handles model loading, metrics, drift detection, and explainability.
"""

import os
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from typing import Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

# Model directory
MODEL_DIR = Path(__file__).parent.parent / "models"

# Check for SHAP and LIME availability
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logger.warning("SHAP not available. Install with: pip install shap")

try:
    import lime
    from lime.lime_tabular import LimeTabularExplainer
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False
    logger.warning("LIME not available. Install with: pip install lime")

# Base feature columns (must match training pipeline order)
BASE_FEATURE_COLS = [
    'peak_ratio',
    'peak_amplitude',
    'rms_amplitude',
    'duration',
    'dominant_frequency',
    'spectral_centroid'
]

# All feature columns (base + derived) - EXACT ORDER from training pipeline
# Training adds: amplitude_ratio, freq_amplitude_product, duration_ratio (in that order)
FEATURE_COLS = BASE_FEATURE_COLS + [
    'amplitude_ratio',        # Added first in training
    'freq_amplitude_product',  # Added second in training
    'duration_ratio'          # Added third in training
]


def get_scaler_feature_order(scaler) -> list:
    """Get the exact feature order expected by the scaler.
    
    The scaler was fit on a DataFrame with this exact order:
    - Base features: peak_ratio, peak_amplitude, rms_amplitude, duration, dominant_frequency, spectral_centroid
    - Derived features: amplitude_ratio, freq_amplitude_product, duration_ratio
    """
    if hasattr(scaler, 'feature_names_in_') and scaler.feature_names_in_ is not None:
        return list(scaler.feature_names_in_)
    # Fallback: return training pipeline order (matches how features were added)
    return [
        'peak_ratio',
        'peak_amplitude',
        'rms_amplitude',
        'duration',
        'dominant_frequency',
        'spectral_centroid',
        'amplitude_ratio',        # Added first in training
        'freq_amplitude_product', # Added second in training
        'duration_ratio'          # Added third in training
    ]


def compute_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute derived features to match training pipeline."""
    df = df.copy()
    
    # Add derived features (same as training pipeline)
    if 'peak_amplitude' in df.columns and 'rms_amplitude' in df.columns:
        df['amplitude_ratio'] = df['peak_amplitude'] / (df['rms_amplitude'] + 1e-10)
    
    if 'dominant_frequency' in df.columns and 'peak_amplitude' in df.columns:
        df['freq_amplitude_product'] = df['dominant_frequency'] * df['peak_amplitude']
    
    if 'duration' in df.columns and 'peak_ratio' in df.columns:
        df['duration_ratio'] = df['duration'] * df['peak_ratio']
    
    # Handle infinite values
    df = df.replace([np.inf, -np.inf], 0)
    
    return df


def load_models() -> Tuple[Optional[object], Optional[object], Optional[object], Optional[object]]:
    """Load trained models and scaler from disk."""
    try:
        # Find latest model files
        model_files = {
            'rf': list(MODEL_DIR.glob('random_forest_*.pkl')),
            'gb': list(MODEL_DIR.glob('gradient_boosting_*.pkl')),
            'lr': list(MODEL_DIR.glob('logistic_regression_*.pkl')),
            'scaler': list(MODEL_DIR.glob('scaler_*.pkl'))
        }

        rf_model = None
        gb_model = None
        lr_model = None
        scaler = None
    
        if model_files['rf']:
            rf_model = joblib.load(sorted(model_files['rf'])[-1])
        if model_files['gb']:
            gb_model = joblib.load(sorted(model_files['gb'])[-1])
        if model_files['lr']:
            lr_model = joblib.load(sorted(model_files['lr'])[-1])
        if model_files['scaler']:
            scaler = joblib.load(sorted(model_files['scaler'])[-1])
        
        return rf_model, gb_model, lr_model, scaler
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        return None, None, None, None


def load_metrics() -> Dict:
    """Load model evaluation metrics."""
    import json
    
    # Try to load from metrics file (check both possible names)
    metrics_file = MODEL_DIR / "model_metrics.json"
    if not metrics_file.exists():
        metrics_file = MODEL_DIR / "metrics.json"
    
    if metrics_file.exists():
        try:
            with open(metrics_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Could not load metrics: {e}")
    
    # Return default structure (will be populated from training)
    return {
        'random_forest': {
            'accuracy': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1_score': 0.0,
            'roc_auc': 0.0
        },
        'gradient_boosting': {
            'accuracy': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1_score': 0.0,
            'roc_auc': 0.0
        },
        'logistic_regression': {
            'accuracy': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1_score': 0.0,
            'roc_auc': 0.0
        }
    }


def load_training_data_sample() -> Optional[pd.DataFrame]:
    """Load a sample of training data for drift detection."""
    # First try to load from saved CSV
    sample_file = MODEL_DIR / "training_data_sample.csv"
    if sample_file.exists():
        try:
            return pd.read_csv(sample_file)
        except Exception as e:
            logger.warning(f"Could not load saved sample: {e}")
    
    # Try to load from Hopsworks
    try:
        from dotenv import load_dotenv
        import hopsworks
        
        env_path = Path(__file__).parent.parent / '.env'
        load_dotenv(env_path)
        API_KEY = os.getenv("HOPSWORKS_API_KEY")
        PROJECT_NAME = os.getenv("HOPSWORKS_PROJECT_NAME", "QuakeAlertWave")
        
        if API_KEY:
            project = hopsworks.login(api_key_value=API_KEY, project=PROJECT_NAME)
            fs = project.get_feature_store()
            wave_fg = fs.get_feature_group(name="waveform_features", version=1)
            df = wave_fg.read()
            
            # Sample for performance
            if len(df) > 1000:
                df = df.sample(n=1000, random_state=42)
            
            return df
    except Exception as e:
        logger.warning(f"Could not load training data from Hopsworks: {e}")
    
    # Return synthetic training data as fallback
    np.random.seed(42)
    return pd.DataFrame({
        'peak_ratio': np.random.uniform(2, 25, 1000),
        'peak_amplitude': np.random.uniform(1e-6, 1e-4, 1000),
        'rms_amplitude': np.random.uniform(5e-7, 5e-5, 1000),
        'duration': np.random.uniform(10, 120, 1000),
        'dominant_frequency': np.random.uniform(0.5, 15, 1000),
        'spectral_centroid': np.random.uniform(1, 8, 1000),
    })


def detect_data_drift(production_data: pd.DataFrame, training_data: pd.DataFrame) -> Dict:
    """Detect data drift between production and training data."""
    drift_results = {}
    
    # Compute derived features for both datasets
    production_data = compute_derived_features(production_data)
    training_data = compute_derived_features(training_data)
    
    for feature in FEATURE_COLS:
        if feature not in production_data.columns or feature not in training_data.columns:
            continue
        
        prod_values = production_data[feature].dropna()
        train_values = training_data[feature].dropna()
        
        if len(prod_values) == 0 or len(train_values) == 0:
            continue
        
        # Calculate statistics
        prod_mean = prod_values.mean()
        train_mean = train_values.mean()
        prod_std = prod_values.std()
        train_std = train_values.std()
        
        # Kolmogorov-Smirnov test for distribution similarity
        from scipy import stats
        try:
            ks_stat, p_value = stats.ks_2samp(prod_values, train_values)
        except:
            ks_stat, p_value = 0.0, 1.0
        
        # Drift score (0-1, higher = more drift)
        mean_diff = abs(prod_mean - train_mean) / (train_std + 1e-10)
        std_diff = abs(prod_std - train_std) / (train_std + 1e-10)
        drift_score = min(1.0, (mean_diff + std_diff + (1 - p_value)) / 3)
        
        drift_results[feature] = {
            'drift_score': drift_score,
            'p_value': p_value,
            'ks_statistic': ks_stat,
            'production_mean': float(prod_mean),
            'training_mean': float(train_mean),
            'production_std': float(prod_std),
            'training_std': float(train_std),
            'has_drift': drift_score > 0.3  # Threshold
            }
    
    return drift_results


def get_shap_explanation(model, data: pd.DataFrame, feature_cols: list) -> Optional[object]:
    """Generate SHAP explanation for model predictions."""
    if not SHAP_AVAILABLE:
        return None
    
    try:
        # Check model type properly
        model_type = type(model).__name__
        
        # Use TreeExplainer for tree-based models
        if 'RandomForest' in model_type or 'GradientBoosting' in model_type or 'DecisionTree' in model_type:
            explainer = shap.TreeExplainer(model)
        # Use LinearExplainer for linear models
        elif 'LogisticRegression' in model_type or 'Linear' in model_type:
            explainer = shap.LinearExplainer(model, data[feature_cols].values)
        # Fallback to KernelExplainer for other models
        else:
            explainer = shap.KernelExplainer(model.predict_proba, data[feature_cols].iloc[:50].values)
        
        # Limit data for performance
        sample_data = data[feature_cols].iloc[:100] if len(data) > 100 else data[feature_cols]
        shap_values = explainer.shap_values(sample_data)
        return shap_values
    except Exception as e:
        logger.error(f"SHAP explanation error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None


def get_lime_explanation(model, data: pd.DataFrame, feature_cols: list, instance_idx: int = 0) -> Optional[Dict]:
    """Generate LIME explanation for a single prediction."""
    if not LIME_AVAILABLE:
        return None
    
    try:
        # Ensure instance_idx is within bounds
        if instance_idx >= len(data):
            instance_idx = len(data) - 1
        if instance_idx < 0:
            instance_idx = 0
        
        # Create explainer with training data
        training_data = data[feature_cols].values
        explainer = LimeTabularExplainer(
            training_data,
            feature_names=feature_cols,
            mode='classification',
            discretize_continuous=True
        )
        
        # Get instance
        instance = data[feature_cols].iloc[instance_idx].values
        
        # Create prediction function
        def predict_fn(X):
            return model.predict_proba(X)
        
        # Generate explanation
        explanation = explainer.explain_instance(
            instance,
            predict_fn, 
            num_features=len(feature_cols),
            top_labels=1
        )
        
        # Extract explanation
        exp_list = explanation.as_list(label=explanation.available_labels()[0])
        
        # Get prediction probability
        pred_proba = model.predict_proba([instance])[0]
        
        return {
            'features': [x[0] for x in exp_list],
            'contributions': [x[1] for x in exp_list],
            'prediction': float(pred_proba[1])  # Probability of class 1
        }
    except Exception as e:
        logger.error(f"LIME explanation error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None


def get_feature_explanations() -> Dict[str, str]:
    """Get detailed human-readable explanations for each feature."""
    return {
        'peak_ratio': """
**What it is:**
The Peak Ratio, also known as the STA/LTA (Short-Term Average / Long-Term Average) ratio, is a fundamental parameter in seismic event detection. It compares the average amplitude of the seismic signal over a short time window (typically 1-5 seconds) to the average amplitude over a longer time window (typically 10-30 seconds).

**How it works:**
- **Short-Term Average (STA)**: Captures recent signal activity, sensitive to sudden changes
- **Long-Term Average (LTA)**: Represents the background noise level
- **Ratio Calculation**: STA ÷ LTA

**What it indicates:**
- **Low values (1-3)**: Normal background noise, no significant seismic activity
- **Medium values (3-10)**: Minor seismic events or noise fluctuations
- **High values (>10)**: Strong seismic signals, likely significant earthquake events
- **Very high values (>50)**: Major seismic events, potentially large earthquakes

**Why it matters:**
This ratio is the primary trigger for earthquake detection algorithms. A sudden increase in the ratio indicates the arrival of seismic waves (P-waves, S-waves), making it crucial for early warning systems.
        """,
        'peak_amplitude': """
**What it is:**
Peak Amplitude represents the maximum displacement or velocity of the ground motion during a seismic event. It measures the strongest point of the seismic wave, typically measured in meters per second (m/s) or meters (m) depending on the sensor type.

**How it's measured:**
- Captured at the moment of maximum ground motion
- Measured from the baseline (zero) to the peak value
- Can be positive (upward motion) or negative (downward motion), but we use the absolute maximum

**What it indicates:**
- **Low amplitude (<0.001 m/s)**: Minor tremors, distant earthquakes, or noise
- **Medium amplitude (0.001-0.01 m/s)**: Local earthquakes, moderate shaking
- **High amplitude (>0.01 m/s)**: Strong local earthquakes, significant ground motion
- **Very high amplitude (>0.1 m/s)**: Major earthquakes, severe shaking

**Why it matters:**
Peak amplitude directly correlates with earthquake magnitude and potential damage. Higher amplitudes indicate stronger ground shaking, which is critical for assessing earthquake impact and issuing appropriate warnings.
        """,
        'rms_amplitude': """
**What it is:**
RMS (Root Mean Square) Amplitude is a statistical measure that represents the average energy of the seismic signal over the entire event duration. Unlike peak amplitude which captures a single moment, RMS amplitude provides a more stable measure of overall signal strength.

**How it's calculated:**
RMS = √(Σ(x²) / n)
Where x represents each sample value and n is the number of samples.

**What it indicates:**
- **Low RMS (<0.0001 m/s)**: Weak signals, background noise, or distant events
- **Medium RMS (0.0001-0.001 m/s)**: Moderate seismic events
- **High RMS (>0.001 m/s)**: Strong seismic events with sustained energy
- **Very high RMS (>0.01 m/s)**: Major earthquakes with prolonged shaking

**Why it matters:**
RMS amplitude gives a better sense of the total energy released during an event, not just the peak. It's less sensitive to single spikes and provides a more reliable measure for event classification. It's particularly useful for distinguishing between brief noise spikes and actual seismic events.
        """,
        'duration': """
**What it is:**
Duration measures the length of time a seismic event lasts, from the initial detection (typically P-wave arrival) until the signal returns to background noise levels. It's measured in seconds.

**How it's determined:**
- **Start**: When the STA/LTA ratio exceeds a threshold (typically 2-3)
- **End**: When the signal amplitude returns to near-baseline levels
- **Measurement**: Time difference between start and end points

**What it indicates:**
- **Short duration (<5 seconds)**: Brief events, noise spikes, or very distant earthquakes
- **Medium duration (5-30 seconds)**: Typical local earthquakes
- **Long duration (30-120 seconds)**: Large earthquakes, multiple phases, or complex events
- **Very long duration (>120 seconds)**: Major earthquakes with extended shaking, aftershocks, or multiple events

**Why it matters:**
Duration is a key indicator of earthquake size and complexity. Longer durations often correlate with larger magnitude earthquakes and can help distinguish between:
- Single-phase events (simple earthquakes)
- Multi-phase events (earthquakes with multiple rupture points)
- Aftershock sequences
- Noise vs. real events (noise is typically very brief)
        """,
        'dominant_frequency': """
**What it is:**
Dominant Frequency is the most prominent frequency component in the seismic signal's frequency spectrum. It represents where most of the signal's energy is concentrated and is measured in Hertz (Hz).

**How it's calculated:**
- The seismic signal is transformed from time domain to frequency domain using Fast Fourier Transform (FFT)
- The frequency with the highest power/amplitude is identified
- This frequency represents the dominant oscillation rate of the ground motion

**What it indicates:**
- **Low frequency (<1 Hz)**: Large earthquakes, surface waves, or distant events
- **Medium frequency (1-10 Hz)**: Typical earthquake frequencies, body waves (P and S waves)
- **High frequency (>10 Hz)**: Local events, noise, or high-frequency ground motion
- **Very high frequency (>20 Hz)**: Usually noise, equipment vibrations, or very local events

**Why it matters:**
Different earthquake types and distances produce different frequency signatures:
- **Large, distant earthquakes**: Lower frequencies (0.1-1 Hz)
- **Local earthquakes**: Higher frequencies (1-10 Hz)
- **Noise**: Very high frequencies (>20 Hz) or random patterns
- **Surface waves**: Lower frequencies than body waves

This helps in event classification and distinguishing real earthquakes from noise or other sources.
        """,
        'spectral_centroid': """
**What it is:**
Spectral Centroid is the "center of mass" of the frequency spectrum. It represents the weighted average frequency, indicating where most of the signal's energy is concentrated. It's measured in Hertz (Hz).

**How it's calculated:**
Spectral Centroid = Σ(f × P(f)) / Σ(P(f))
Where f is frequency and P(f) is the power at that frequency.

**What it indicates:**
- **Low centroid (<2 Hz)**: Energy concentrated at low frequencies - typical of large or distant earthquakes
- **Medium centroid (2-8 Hz)**: Balanced frequency distribution - typical of moderate local earthquakes
- **High centroid (>8 Hz)**: Energy concentrated at high frequencies - may indicate noise, local events, or high-frequency ground motion

**Why it matters:**
Spectral centroid provides a single number that summarizes the frequency distribution:
- **Lower values**: Suggest larger magnitude or more distant events
- **Higher values**: Suggest local events, noise, or high-frequency content
- **Stable values**: Consistent frequency content (real event)
- **Variable values**: Changing frequency content (may indicate noise or complex event)

It's particularly useful for distinguishing between different types of seismic sources and filtering out high-frequency noise.
        """,
        'amplitude_ratio': """
**What it is:**
Amplitude Ratio is a derived feature that compares the peak amplitude to the RMS amplitude. It's calculated as: Peak Amplitude ÷ RMS Amplitude.

**How it's calculated:**
Amplitude Ratio = Peak Amplitude / (RMS Amplitude + ε)
Where ε is a small constant (1e-10) to prevent division by zero.

**What it indicates:**
- **Low ratio (1-2)**: Signal is relatively uniform, consistent energy throughout
- **Medium ratio (2-5)**: Moderate peak-to-average ratio, typical of most earthquakes
- **High ratio (5-10)**: Strong peak with relatively lower average - indicates sharp, impulsive events
- **Very high ratio (>10)**: Very sharp spike - may indicate noise, equipment glitch, or very impulsive event

**Why it matters:**
This ratio helps distinguish between:
- **Real earthquakes**: Usually have moderate ratios (2-5) with sustained energy
- **Noise spikes**: Often have very high ratios (single sharp spike)
- **Complex events**: May have varying ratios depending on phase
- **Signal quality**: Very high ratios might indicate data quality issues

It's essentially a signal-to-noise indicator that helps validate event quality and distinguish real seismic events from artifacts.
        """,
        'duration_ratio': """
**What it is:**
Duration Ratio is a derived feature that combines temporal and amplitude characteristics. It's calculated as: Duration × Peak Ratio.

**How it's calculated:**
Duration Ratio = Duration × Peak Ratio
This multiplies the event duration (in seconds) by the STA/LTA peak ratio.

**What it indicates:**
- **Low ratio (<50)**: Short events or weak signals - likely noise or minor events
- **Medium ratio (50-200)**: Typical earthquake characteristics
- **High ratio (200-500)**: Strong, sustained events - likely significant earthquakes
- **Very high ratio (>500)**: Major events with both high amplitude and long duration

**Why it matters:**
This feature captures both temporal and amplitude characteristics simultaneously:
- **High duration + High peak ratio**: Strong, sustained earthquake - high confidence event
- **High duration + Low peak ratio**: Long but weak signal - may be noise or distant event
- **Low duration + High peak ratio**: Sharp spike - may be noise or very local event
- **Low duration + Low peak ratio**: Weak, brief signal - likely noise

It helps the model distinguish between:
- Real earthquakes (balanced duration and amplitude)
- Noise (often brief spikes or long weak signals)
- Different earthquake magnitudes (larger events tend to have higher ratios)
        """,
        'freq_amplitude_product': """
**What it is:**
Frequency-Amplitude Product is a derived feature that combines frequency and amplitude information. It's calculated as: Dominant Frequency × Peak Amplitude.

**How it's calculated:**
Freq-Amplitude Product = Dominant Frequency × Peak Amplitude
This multiplies the dominant frequency (Hz) by the peak amplitude (m/s).

**What it indicates:**
- **Low product (<0.01)**: Low frequency and/or low amplitude - distant or small events
- **Medium product (0.01-0.1)**: Typical earthquake characteristics
- **High product (0.1-1.0)**: High frequency and/or high amplitude - local, strong events
- **Very high product (>1.0)**: Very strong local events or potential noise/artifacts

**Why it matters:**
This feature captures the energy-frequency relationship:
- **High frequency + High amplitude**: Strong local events - high confidence earthquake
- **Low frequency + High amplitude**: Large, possibly distant earthquakes
- **High frequency + Low amplitude**: Local but weak events or noise
- **Low frequency + Low amplitude**: Distant or very small events

It helps the model understand:
- **Event distance**: Local events have higher frequencies
- **Event magnitude**: Larger events have higher amplitudes
- **Event type**: Different earthquake types have different frequency-amplitude relationships
- **Noise filtering**: Noise often has unusual frequency-amplitude combinations

This combined feature provides richer information than frequency or amplitude alone, helping improve classification accuracy.
        """
    }


def save_training_data_sample(df: pd.DataFrame, max_samples: int = 1000) -> bool:
    """Save a sample of training data for drift detection."""
    try:
        sample_file = MODEL_DIR / "training_data_sample.csv"
        sample_df = df.sample(n=min(max_samples, len(df)), random_state=42) if len(df) > max_samples else df
        sample_df.to_csv(sample_file, index=False)
        return True
    except Exception as e:
        logger.error(f"Error saving training data sample: {e}")
        return False
