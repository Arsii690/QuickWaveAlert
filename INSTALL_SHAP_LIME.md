# Installing SHAP and LIME for QuakeAlertWave

## ✅ Compatible Versions

For **Python 3.11** with your project dependencies:
- **SHAP**: `0.43.0` (compatible with numpy<2.0 and scikit-learn 1.3.2)
- **LIME**: `0.2.0.1` (compatible with Python 3.11 and scikit-learn 1.3.2)

## 📦 Installation Command

```bash
# Activate your virtual environment first
source venv/bin/activate

# Install compatible versions
pip install shap==0.43.0 lime==0.2.0.1
```

## 🔍 Why These Versions?

- **SHAP 0.43.0**: 
  - ✅ Works with numpy < 2.0.0 (your constraint)
  - ✅ Compatible with scikit-learn 1.3.2
  - ✅ Supports Python 3.11
  - ⚠️ Newer versions (0.44.0+) may have numpy 2.0 compatibility issues

- **LIME 0.2.0.1**:
  - ✅ Latest stable version
  - ✅ Fully compatible with Python 3.11
  - ✅ Works with scikit-learn 1.3.2

## ✅ Verification

After installation, verify both packages:

```bash
# Check SHAP (has __version__ attribute)
python -c "import shap; print(f'✅ SHAP version: {shap.__version__}')"

# Check LIME (doesn't have __version__, but we can verify import)
python -c "import lime; from lime.lime_tabular import LimeTabularExplainer; print('✅ LIME installed successfully!')"
```

Expected output:
```
✅ SHAP version: 0.43.0
✅ LIME installed successfully!
```

**Note:** LIME doesn't expose a `__version__` attribute, but if the import succeeds, it's installed correctly.

## 🚨 Troubleshooting

If you encounter issues:

1. **NumPy compatibility error**: Make sure numpy < 2.0.0
   ```bash
   pip install "numpy>=1.24.0,<2.0.0"
   ```

2. **Scikit-learn compatibility**: Ensure scikit-learn 1.3.2 is installed
   ```bash
   pip install scikit-learn==1.3.2
   ```

3. **Reinstall if needed**:
   ```bash
   pip uninstall shap lime
   pip install shap==0.43.0 lime==0.2.0.1
   ```

## 📝 Alternative: Install from requirements.txt

You can also uncomment the lines in `requirements.txt`:

```bash
# Edit requirements.txt and uncomment:
# shap==0.43.0
# lime==0.2.0.1

# Then install:
pip install -r requirements.txt
```

