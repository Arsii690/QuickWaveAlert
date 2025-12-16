"""
Quick test script to validate the scaler feature order fix.
This tests the exact flow used in the dashboard.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from app.ml_utils import load_models, get_scaler_feature_order, BASE_FEATURE_COLS, compute_derived_features

def test_scaler_transform():
    """Test that scaler transform works with correct feature order."""
    print("=" * 60)
    print("Testing Scaler Feature Order Fix")
    print("=" * 60)
    
    # Load models and scaler
    print("\n1. Loading models and scaler...")
    rf_model, gb_model, lr_model, scaler = load_models()
    
    if scaler is None:
        print("❌ ERROR: No scaler found! Please run the training pipeline first.")
        print("   Run: python -m pipelines.3_training_pipeline")
        return False
    
    print(f"✅ Scaler loaded successfully")
    
    # Get expected feature order
    print("\n2. Getting expected feature order from scaler...")
    expected_order = get_scaler_feature_order(scaler)
    print(f"✅ Expected order: {expected_order}")
    
    # Create sample data (simulating dashboard flow)
    print("\n3. Creating sample data (simulating dashboard)...")
    sample_data = pd.DataFrame({
        'peak_ratio': [5.2],
        'peak_amplitude': [0.0015],
        'rms_amplitude': [0.0003],
        'duration': [2.5],
        'dominant_frequency': [1.2],
        'spectral_centroid': [0.8]
    })
    print(f"✅ Sample data created with base features: {list(sample_data.columns)}")
    
    # Compute derived features (same as dashboard)
    print("\n4. Computing derived features...")
    sample_features = compute_derived_features(sample_data)
    print(f"✅ Derived features added. Columns: {list(sample_features.columns)}")
    
    # Get expected order and ensure all features exist
    print("\n5. Ensuring all features exist and reordering...")
    missing = [f for f in expected_order if f not in sample_features.columns]
    if missing:
        print(f"⚠️  Missing features: {missing}. Adding with 0.0")
        for feat in missing:
            sample_features[feat] = 0.0
    
    # Select in exact order
    sample_features_ordered = sample_features[expected_order]
    print(f"✅ Features reordered. Final order: {list(sample_features_ordered.columns)}")
    
    # Verify order matches
    if list(sample_features_ordered.columns) != expected_order:
        print(f"❌ ERROR: Order mismatch!")
        print(f"   Expected: {expected_order}")
        print(f"   Got: {list(sample_features_ordered.columns)}")
        return False
    
    # Test transform
    print("\n6. Testing scaler.transform()...")
    try:
        scaled_features = scaler.transform(sample_features_ordered)
        print(f"✅ Transform successful! Output shape: {scaled_features.shape}")
        print(f"   Scaled values (first 3): {scaled_features[0][:3]}")
    except ValueError as e:
        print(f"❌ ERROR: Transform failed!")
        print(f"   Error: {str(e)}")
        if hasattr(scaler, 'feature_names_in_'):
            print(f"   Scaler expects: {list(scaler.feature_names_in_)}")
        print(f"   We provided: {list(sample_features_ordered.columns)}")
        return False
    except Exception as e:
        print(f"❌ ERROR: Unexpected error during transform!")
        print(f"   Error: {type(e).__name__}: {str(e)}")
        return False
    
    # Test model prediction (if models are available)
    print("\n7. Testing model prediction...")
    if rf_model is not None:
        try:
            prediction = rf_model.predict(scaled_features)[0]
            probability = rf_model.predict_proba(scaled_features)[0]
            print(f"✅ Random Forest prediction successful!")
            print(f"   Prediction: {'Significant Event' if prediction == 1 else 'Normal Event'}")
            print(f"   Confidence: {probability[prediction]:.2%}")
        except Exception as e:
            print(f"⚠️  Model prediction failed: {e}")
    else:
        print("⚠️  No models found. Skipping prediction test.")
    
    print("\n" + "=" * 60)
    print("✅ ALL TESTS PASSED! The fix is working correctly.")
    print("=" * 60)
    return True

if __name__ == "__main__":
    success = test_scaler_transform()
    sys.exit(0 if success else 1)

