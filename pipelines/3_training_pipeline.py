"""
ML Training Pipeline for Seismic Event Classification
Trains three models to predict significant seismic events from waveform features.
"""

import os
import sys
import hopsworks
import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
from dotenv import load_dotenv
from datetime import datetime
import logging
from typing import Dict, Tuple

# ML libraries
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score
)
import warnings
warnings.filterwarnings('ignore')

# Prefect
from prefect import flow, task
from prefect.logging import get_run_logger

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment
env_path = Path(__file__).parent.parent / '.env'
load_dotenv(dotenv_path=env_path)

API_KEY = os.getenv("HOPSWORKS_API_KEY")
PROJECT_NAME = os.getenv("HOPSWORKS_PROJECT_NAME", "QuakeAlertWave")

# Model save directory
MODEL_DIR = Path(__file__).parent.parent / "models"
MODEL_DIR.mkdir(exist_ok=True)


def generate_synthetic_training_data(n_samples: int = 1000) -> pd.DataFrame:
    """Generate synthetic seismic data for training when Hopsworks data is unavailable."""
    np.random.seed(42)
    
    # Create realistic synthetic data
    df = pd.DataFrame({
        'timestamp': [int(1735000000000 + i * 1000) for i in range(n_samples)],
        'network': np.random.choice(['IU', 'US'], n_samples),
        'station': np.random.choice(['ANMO', 'COLA', 'TAU', 'MAJO'], n_samples),
        'channel': ['BHZ'] * n_samples,
        'location': ['00'] * n_samples,
        'peak_ratio': np.random.uniform(2, 25, n_samples),
        'duration': np.random.uniform(10, 120, n_samples),
        'peak_amplitude': np.random.uniform(1e-6, 1e-4, n_samples),
        'rms_amplitude': np.random.uniform(5e-7, 5e-5, n_samples),
        'dominant_frequency': np.random.uniform(0.5, 15, n_samples),
        'spectral_centroid': np.random.uniform(1, 8, n_samples),
    })
    
    # Add derived features
    df['is_significant'] = (df['peak_ratio'] > 10.0).astype(int)
    df['magnitude_estimate'] = df['peak_amplitude'].apply(
        lambda x: min(9.0, 2.0 + 2.0 * np.log10(abs(x))) if x > 0 else 0.0
    )
    
    return df


@task(name="Load Data from Hopsworks")
def load_data_from_hopsworks() -> pd.DataFrame:
    """Load waveform features from Hopsworks Feature Store or generate synthetic data."""
    logger = get_run_logger()
    
    # Try to load from Hopsworks first
    if API_KEY:
        try:
            logger.info("Connecting to Hopsworks...")
            project = hopsworks.login(api_key_value=API_KEY, project=PROJECT_NAME)
            fs = project.get_feature_store()
            
            logger.info("Loading waveform_features feature group...")
            
            # Try to get the feature group
            wave_fg = None
            try:
                wave_fg = fs.get_feature_group(name="waveform_features", version=1)
            except Exception as fg_error:
                logger.warning(f"Could not get feature group: {fg_error}")
            
            # Try to read data if feature group exists
            if wave_fg is not None and hasattr(wave_fg, 'read'):
                try:
                    logger.info("✅ Feature group found, reading data...")
                    df = wave_fg.read()
                    
                    if df is not None and not df.empty:
                        logger.info(f"✅ Loaded {len(df)} samples from Hopsworks")
                        return df
                    else:
                        logger.warning("Feature group is empty or data not yet materialized")
                except Exception as read_error:
                    logger.warning(f"Could not read from Hopsworks: {read_error}")
                    logger.info("Data may still be materializing. Falling back to synthetic data.")
            
        except Exception as e:
            logger.warning(f"Hopsworks connection failed: {e}")
    else:
        logger.warning("No Hopsworks API key found")
    
    # Fallback: Generate synthetic training data
    logger.info("⚠️ Using synthetic training data as fallback")
    logger.info("  (Run backfill.py and wait for materialization to use real data)")
    
    df = generate_synthetic_training_data(n_samples=1000)
    logger.info(f"✅ Generated {len(df)} synthetic samples for training")
    
    return df


@task(name="Prepare Features and Target")
def prepare_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, list]:
    """
    Prepare features and target variable for training.
    
    Target: is_significant (binary classification)
    - 1: Significant event (peak_ratio > 10.0)
    - 0: Normal event
    """
    logger = get_run_logger()
    
    logger.info("Preparing features and target...")
    
    # Feature columns
    feature_cols = [
        'peak_ratio',
        'peak_amplitude',
        'rms_amplitude',
        'duration',
        'dominant_frequency',
        'spectral_centroid'
    ]
    
    # Check if all features exist
    missing_cols = [col for col in feature_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing feature columns: {missing_cols}")
    
    # Extract features
    X = df[feature_cols].copy()
    
    # Create or use target variable
    if 'is_significant' not in df.columns:
        # Create target based on peak_ratio threshold
        y = (df['peak_ratio'] > 10.0).astype(int)
        logger.info("Created 'is_significant' target from peak_ratio")
    else:
        y = df['is_significant'].copy()
    
    # Handle missing values
    X = X.fillna(0)
    
    # Remove infinite values
    X = X.replace([np.inf, -np.inf], 0)
    
    # Feature engineering: Add derived features
    X['amplitude_ratio'] = X['peak_amplitude'] / (X['rms_amplitude'] + 1e-10)
    X['freq_amplitude_product'] = X['dominant_frequency'] * X['peak_amplitude']
    X['duration_ratio'] = X['duration'] * X['peak_ratio']
    
    logger.info(f"Feature matrix shape: {X.shape}")
    logger.info(f"Target distribution:\n{y.value_counts()}")
    logger.info(f"Class balance: {y.value_counts(normalize=True).to_dict()}")
    
    return X, y, feature_cols


@task(name="Split and Scale Data")
def split_and_scale_data(X: pd.DataFrame, y: pd.Series) -> Tuple:
    """Split data into train/test sets and scale features."""
    logger = get_run_logger()
    
    # Split data (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    logger.info(f"Train set: {len(X_train)} samples")
    logger.info(f"Test set: {len(X_test)} samples")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert back to DataFrame for feature names
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


@task(name="Train Random Forest Model")
def train_random_forest(X_train: pd.DataFrame, y_train: pd.Series) -> RandomForestClassifier:
    """Train Random Forest Classifier."""
    logger = get_run_logger()
    
    logger.info("Training Random Forest Classifier...")
    
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    # Cross-validation score
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    logger.info(f"✅ Random Forest - CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    
    return model


@task(name="Train Gradient Boosting Model")
def train_gradient_boosting(X_train: pd.DataFrame, y_train: pd.Series) -> GradientBoostingClassifier:
    """Train Gradient Boosting Classifier."""
    logger = get_run_logger()
    
    logger.info("Training Gradient Boosting Classifier...")
    
    model = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    # Cross-validation score
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    logger.info(f"✅ Gradient Boosting - CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    
    return model


@task(name="Train Logistic Regression Model")
def train_logistic_regression(X_train: pd.DataFrame, y_train: pd.Series) -> LogisticRegression:
    """Train Logistic Regression (baseline model)."""
    logger = get_run_logger()
    
    logger.info("Training Logistic Regression...")
    
    model = LogisticRegression(
        max_iter=1000,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    # Cross-validation score
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    logger.info(f"✅ Logistic Regression - CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    
    return model


@task(name="Evaluate Model")
def evaluate_model(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str
) -> Dict:
    """Evaluate model performance on test set."""
    logger = get_run_logger()
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Evaluating {model_name}")
    logger.info(f"{'='*60}")
    
    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    logger.info(f"Accuracy:  {accuracy:.4f}")
    logger.info(f"Precision: {precision:.4f}")
    logger.info(f"Recall:    {recall:.4f}")
    logger.info(f"F1 Score:  {f1:.4f}")
    logger.info(f"ROC AUC:   {roc_auc:.4f}")
    
    # Classification report
    logger.info(f"\nClassification Report:")
    logger.info(f"\n{classification_report(y_test, y_pred)}")
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    logger.info(f"\nConfusion Matrix:")
    logger.info(f"\n{cm}")
    
    metrics = {
        'model_name': model_name,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': roc_auc,
        'confusion_matrix': cm.tolist()
    }
    
    return metrics


@task(name="Save Model Locally")
def save_model_locally(model, scaler, model_name: str) -> Tuple[Path, Path]:
    """Save model and scaler to local disk."""
    logger = get_run_logger()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_filename = f"{model_name}_{timestamp}.pkl"
    scaler_filename = f"scaler_{timestamp}.pkl"
    
    model_path = MODEL_DIR / model_filename
    scaler_path = MODEL_DIR / scaler_filename
    
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    
    logger.info(f"✅ Model saved: {model_path}")
    logger.info(f"✅ Scaler saved: {scaler_path}")
    
    return model_path, scaler_path


@task(name="Upload Model to Hopsworks")
def upload_to_hopsworks(
    model_path: Path,
    scaler_path: Path,
    model_name: str,
    metrics: Dict,
    feature_cols: list
):
    """Upload trained model to Hopsworks Model Registry."""
    logger = get_run_logger()
    
    try:
        logger.info("Connecting to Hopsworks Model Registry...")
        project = hopsworks.login(api_key_value=API_KEY, project=PROJECT_NAME)
        mr = project.get_model_registry()
        
        # Create model schema (input/output) - IMPORTANT: Must specify 'name' and 'type' for each feature
        from hsml.schema import Schema
        from hsml.model_schema import ModelSchema
        
        # Build proper input schema with name AND type for each feature
        input_schema_list = [{"name": col, "type": "float"} for col in feature_cols]
        input_schema = Schema(input_schema_list)
        output_schema = Schema([{"name": "is_significant", "type": "int"}])
        model_schema = ModelSchema(input_schema=input_schema, output_schema=output_schema)
        
        # Prepare metrics for Hopsworks (only numeric values allowed)
        clean_metrics = {
            "accuracy": float(metrics.get("accuracy", 0)),
            "precision": float(metrics.get("precision", 0)),
            "recall": float(metrics.get("recall", 0)),
            "f1_score": float(metrics.get("f1_score", 0)),
            "roc_auc": float(metrics.get("roc_auc", 0))
        }
        
        # Register model
        logger.info(f"Uploading {model_name} to Hopsworks Model Registry...")
        
        model = mr.sklearn.create_model(
            name=f"seismic_{model_name}",
            description=f"Seismic event classifier using {model_name} - Detects significant seismic events",
            metrics=clean_metrics,
            model_schema=model_schema
        )
        
        # Save model file to Hopsworks
        model.save(str(model_path))
        
        logger.info(f"✅ Model 'seismic_{model_name}' uploaded to Hopsworks Model Registry!")
        logger.info(f"   Version: {model.version}")
        logger.info(f"   Metrics: Accuracy={clean_metrics['accuracy']:.4f}, F1={clean_metrics['f1_score']:.4f}")
        
        return True
        
    except Exception as e:
        logger.warning(f"⚠️ Failed to upload to Hopsworks Model Registry: {e}")
        logger.info("💾 Model is saved locally and can be used from disk")
        return False


@flow(name="ML Training Pipeline")
def training_pipeline():
    """
    Main training pipeline:
    1. Load data from Hopsworks
    2. Prepare features
    3. Train three models
    4. Evaluate and compare
    5. Save best model
    """
    logger = get_run_logger()
    
    logger.info("="*60)
    logger.info("🚀 Starting ML Training Pipeline")
    logger.info("="*60)
    
    # Load data
    df = load_data_from_hopsworks()
    
    if len(df) < 100:
        logger.warning(f"⚠️ Only {len(df)} samples available. Recommend at least 1000 samples for reliable training.")
        logger.warning("⚠️ Run backfill.py to collect more data first.")
        return
    
    # Prepare features
    X, y, feature_cols = prepare_features(df)
    
    # Split and scale
    X_train, X_test, y_train, y_test, scaler = split_and_scale_data(X, y)
    
    # Train models
    logger.info("\n🎯 Training Models...")
    rf_model = train_random_forest(X_train, y_train)
    gb_model = train_gradient_boosting(X_train, y_train)
    lr_model = train_logistic_regression(X_train, y_train)
    
    # Evaluate models
    logger.info("\n📊 Evaluating Models...")
    rf_metrics = evaluate_model(rf_model, X_test, y_test, "RandomForest")
    gb_metrics = evaluate_model(gb_model, X_test, y_test, "GradientBoosting")
    lr_metrics = evaluate_model(lr_model, X_test, y_test, "LogisticRegression")
    
    # Compare models
    logger.info("\n" + "="*60)
    logger.info("📈 MODEL COMPARISON")
    logger.info("="*60)
    
    comparison = pd.DataFrame([
        {
            'Model': rf_metrics['model_name'],
            'Accuracy': rf_metrics['accuracy'],
            'Precision': rf_metrics['precision'],
            'Recall': rf_metrics['recall'],
            'F1 Score': rf_metrics['f1_score'],
            'ROC AUC': rf_metrics['roc_auc']
        },
        {
            'Model': gb_metrics['model_name'],
            'Accuracy': gb_metrics['accuracy'],
            'Precision': gb_metrics['precision'],
            'Recall': gb_metrics['recall'],
            'F1 Score': gb_metrics['f1_score'],
            'ROC AUC': gb_metrics['roc_auc']
        },
        {
            'Model': lr_metrics['model_name'],
            'Accuracy': lr_metrics['accuracy'],
            'Precision': lr_metrics['precision'],
            'Recall': lr_metrics['recall'],
            'F1 Score': lr_metrics['f1_score'],
            'ROC AUC': lr_metrics['roc_auc']
        }
    ])
    
    logger.info(f"\n{comparison.to_string(index=False)}")
    
    # Find best model
    best_idx = comparison['F1 Score'].idxmax()
    best_model_name = comparison.loc[best_idx, 'Model']
    logger.info(f"\n🏆 Best Model: {best_model_name}")
    
    # Save all models locally
    logger.info("\n💾 Saving Models...")
    rf_path, scaler_path = save_model_locally(rf_model, scaler, "random_forest")
    gb_path, _ = save_model_locally(gb_model, scaler, "gradient_boosting")
    lr_path, _ = save_model_locally(lr_model, scaler, "logistic_regression")
    
    # Save metrics to JSON for dashboard
    import json
    metrics_dict = {
        'random_forest': rf_metrics,
        'gradient_boosting': gb_metrics,
        'logistic_regression': lr_metrics
    }
    metrics_file = MODEL_DIR / "model_metrics.json"
    with open(metrics_file, 'w') as f:
        json.dump(metrics_dict, f, indent=2)
    logger.info(f"✅ Metrics saved to: {metrics_file}")
    
    # Save training data sample for drift detection
    try:
        from app.ml_utils import save_training_data_sample
        if save_training_data_sample(df, max_samples=1000):
            logger.info("✅ Training data sample saved for drift detection")
    except Exception as e:
        logger.warning(f"Could not save training data sample: {e}")
    
    # Upload best model to Hopsworks
    if best_model_name == "RandomForest":
        upload_to_hopsworks(rf_path, scaler_path, "random_forest", rf_metrics, feature_cols)
    elif best_model_name == "GradientBoosting":
        upload_to_hopsworks(gb_path, scaler_path, "gradient_boosting", gb_metrics, feature_cols)
    else:
        upload_to_hopsworks(lr_path, scaler_path, "logistic_regression", lr_metrics, feature_cols)
    
    logger.info("\n" + "="*60)
    logger.info("✅ Training Pipeline Complete!")
    logger.info("="*60)
    logger.info(f"Models saved in: {MODEL_DIR}")
    logger.info(f"Best model uploaded to Hopsworks: {best_model_name}")


if __name__ == "__main__":
    training_pipeline()

