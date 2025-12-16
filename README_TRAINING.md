# 🎓 ML Training Pipeline Guide

## Overview

The training pipeline (`pipelines/3_training_pipeline.py`) trains three machine learning models to classify significant seismic events from waveform features.

## Models

### 1. **Random Forest Classifier**
- **Purpose**: Ensemble learning with decision trees
- **Pros**: High accuracy, handles non-linear patterns, feature importance
- **Training time**: ~10-30 seconds (100 estimators)
- **Best for**: Balanced datasets with complex patterns

### 2. **Gradient Boosting Classifier**
- **Purpose**: Sequential ensemble learning
- **Pros**: Often highest accuracy, good with imbalanced data
- **Training time**: ~15-45 seconds (100 estimators)
- **Best for**: Maximizing performance metrics

### 3. **Logistic Regression**
- **Purpose**: Linear baseline model
- **Pros**: Very fast, interpretable, good baseline
- **Training time**: ~1-5 seconds
- **Best for**: Quick iterations, explainability

## What the Models Predict

**Target**: `is_significant` (Binary Classification)
- **Class 1**: Significant event (peak_ratio > 10.0) - Likely earthquake
- **Class 0**: Normal event - Background seismic noise

**Input Features**:
- `peak_ratio` - STA/LTA ratio
- `peak_amplitude` - Maximum amplitude
- `rms_amplitude` - RMS amplitude
- `duration` - Event duration
- `dominant_frequency` - Primary frequency
- `spectral_centroid` - Spectral center

**Derived Features** (created during training):
- `amplitude_ratio` - Peak/RMS ratio
- `freq_amplitude_product` - Frequency × Amplitude
- `duration_ratio` - Duration × Peak ratio

## Prerequisites

### 1. Data Requirements
- **Minimum**: 100 samples (will show warning)
- **Recommended**: 1,000+ samples for reliable training
- **Optimal**: 10,000+ samples for production-ready models

Run backfill first if needed:
```bash
python backfill.py  # Loads 30 days of historical data
```

### 2. Environment Setup
```bash
cd /Users/kainat/Desktop/QuakeAlertWave
source venv/bin/activate
```

## Running the Training Pipeline

### Basic Usage
```bash
python pipelines/3_training_pipeline.py
```

### With Prefect (Orchestrated)
```bash
prefect server start  # Terminal 1
python pipelines/3_training_pipeline.py  # Terminal 2
```

## Pipeline Steps

1. **Load Data** - Fetches features from Hopsworks
2. **Prepare Features** - Creates feature matrix and target
3. **Split & Scale** - 80/20 train/test split, standardization
4. **Train Models** - Trains all 3 models in parallel
5. **Evaluate** - Computes metrics on test set
6. **Compare** - Ranks models by F1 score
7. **Save** - Saves all models locally
8. **Upload** - Uploads best model to Hopsworks

## Expected Output

```
==========================================
🚀 Starting ML Training Pipeline
==========================================
Connecting to Hopsworks...
✅ Loaded 5432 samples from Hopsworks

Preparing features and target...
Feature matrix shape: (5432, 9)
Target distribution:
0    4523
1     909
Class balance: {0: 0.83, 1: 0.17}

Train set: 4345 samples
Test set: 1087 samples

🎯 Training Models...
✅ Random Forest - CV Accuracy: 0.9234 (+/- 0.0156)
✅ Gradient Boosting - CV Accuracy: 0.9312 (+/- 0.0142)
✅ Logistic Regression - CV Accuracy: 0.8876 (+/- 0.0198)

📊 Evaluating Models...

============================================================
Evaluating RandomForest
============================================================
Accuracy:  0.9245
Precision: 0.8823
Recall:    0.8567
F1 Score:  0.8693
ROC AUC:   0.9621

============================================================
Evaluating GradientBoosting
============================================================
Accuracy:  0.9334
Precision: 0.9012
Recall:    0.8734
F1 Score:  0.8871
ROC AUC:   0.9702

============================================================
Evaluating LogisticRegression
============================================================
Accuracy:  0.8912
Precision: 0.7845
Recall:    0.8123
F1 Score:  0.7982
ROC AUC:   0.9234

📈 MODEL COMPARISON
============================================================
           Model  Accuracy  Precision  Recall  F1 Score  ROC AUC
    RandomForest    0.9245     0.8823  0.8567    0.8693   0.9621
GradientBoosting    0.9334     0.9012  0.8734    0.8871   0.9702
LogisticRegression  0.8912     0.7845  0.8123    0.7982   0.9234

🏆 Best Model: GradientBoosting

💾 Saving Models...
✅ Model saved: models/random_forest_20251215_153045.pkl
✅ Scaler saved: models/scaler_20251215_153045.pkl
✅ Model saved: models/gradient_boosting_20251215_153045.pkl
✅ Model saved: models/logistic_regression_20251215_153045.pkl

Uploading GradientBoosting to Hopsworks...
✅ Model 'gradient_boosting' uploaded to Hopsworks Model Registry

============================================================
✅ Training Pipeline Complete!
============================================================
Models saved in: /Users/kainat/Desktop/QuakeAlertWave/models
Best model uploaded to Hopsworks: GradientBoosting
```

## Performance Expectations

### Typical Metrics (with 5000+ samples)

| Model | Accuracy | Precision | Recall | F1 Score | Training Time |
|-------|----------|-----------|--------|----------|---------------|
| **Random Forest** | 90-93% | 85-90% | 83-88% | 85-89% | 10-30s |
| **Gradient Boosting** | 92-95% | 88-93% | 85-90% | 87-91% | 15-45s |
| **Logistic Regression** | 87-91% | 75-82% | 78-85% | 77-83% | 1-5s |

### Why These Models?

✅ **High Accuracy**: 90%+ accuracy expected  
✅ **Fast Training**: <1 minute on standard hardware  
✅ **Aligned to Project**: Binary classification for seismic events  
✅ **Production-Ready**: Robust, well-tested algorithms  
✅ **Interpretable**: Feature importance available  

## Output Files

### Local Models
```
models/
├── random_forest_YYYYMMDD_HHMMSS.pkl
├── gradient_boosting_YYYYMMDD_HHMMSS.pkl
├── logistic_regression_YYYYMMDD_HHMMSS.pkl
└── scaler_YYYYMMDD_HHMMSS.pkl
```

### Hopsworks Model Registry
- Best model uploaded automatically
- Includes metrics, schema, and versioning
- Accessible via Hopsworks UI

## Using Trained Models

### Load from Local File
```python
import joblib

# Load model and scaler
model = joblib.load('models/gradient_boosting_20251215_153045.pkl')
scaler = joblib.load('models/scaler_20251215_153045.pkl')

# Predict
features = [[15.2, 1.2e-5, 8.3e-6, 45.0, 2.5, 3.1]]
features_scaled = scaler.transform(features)
prediction = model.predict(features_scaled)
probability = model.predict_proba(features_scaled)

print(f"Prediction: {'Significant' if prediction[0] == 1 else 'Normal'}")
print(f"Probability: {probability[0][1]:.2%}")
```

### Load from Hopsworks
```python
import hopsworks

project = hopsworks.login(api_key_value=API_KEY)
mr = project.get_model_registry()

# Get latest version
model = mr.get_model("gradient_boosting", version=1)
model_dir = model.download()

# Load and predict
import joblib
loaded_model = joblib.load(f"{model_dir}/gradient_boosting.pkl")
```

## Troubleshooting

### Issue: Not Enough Data
```
⚠️ Only 50 samples available. Recommend at least 1000 samples.
```
**Solution**: Run `python backfill.py` to collect more historical data

### Issue: Hopsworks Connection Failed
```
Failed to load data from Hopsworks: Could not find feature group
```
**Solution**: 
1. Check `.env` file has `HOPSWORKS_API_KEY`
2. Verify feature group exists: Run `python pipelines/2_waveform_pipeline.py` first

### Issue: Imbalanced Classes
```
Class balance: {0: 0.98, 1: 0.02}
```
**Solution**: This is normal for seismic data (few significant events). Models handle this automatically.

### Issue: Low Accuracy
```
Accuracy: 0.65
```
**Solution**:
1. Collect more data (run backfill for full 30 days)
2. Check data quality in Hopsworks
3. Adjust STA/LTA threshold in detection

## Next Steps

1. **Integrate with API**: Update `app/main.py` to use trained models
2. **Real-time Predictions**: Add model inference to waveform analysis
3. **Model Monitoring**: Track prediction accuracy over time
4. **Retraining**: Schedule periodic retraining with new data

## Model Performance Tips

### To Improve Accuracy
- Collect more training data (run backfill for longer periods)
- Add more features (velocity, acceleration, etc.)
- Tune hyperparameters (grid search)
- Use XGBoost instead of GradientBoost

### To Reduce Training Time
- Use Logistic Regression for quick iterations
- Reduce `n_estimators` in ensemble models
- Use smaller train/test split

### To Handle Imbalanced Data
- Current models already handle class imbalance
- For extreme cases: use SMOTE (oversampling)
- Adjust class weights in model parameters

## Summary

✅ **3 ML models**: Random Forest, Gradient Boosting, Logistic Regression  
✅ **High accuracy**: 90-95% expected  
✅ **Fast training**: <1 minute  
✅ **Auto-comparison**: Best model selected and uploaded  
✅ **Production-ready**: Saved locally and in Hopsworks  

**Run now**: `python pipelines/3_training_pipeline.py`

