# 🚀 Deployment Guide

## Quick Start

### 1. Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Start API
uvicorn app.main:app --reload

# Start Dashboard (separate terminal)
streamlit run dashboard.py
```

### 2. Docker Deployment

```bash
# Build image
docker build -t quakealert:latest .

# Run container
docker run -p 8000:8000 quakealert:latest
```

### 3. Hugging Face Spaces Deployment

#### Step 1: Create Space
1. Go to [Hugging Face Spaces](https://huggingface.co/spaces)
2. Click "Create new Space"
3. Choose:
   - **SDK**: Docker
   - **Name**: quakealert (or your choice)
   - **Visibility**: Public or Private

#### Step 2: Configure Space
1. Upload your code to the Space (or connect GitHub repo)
2. Add secrets in Space settings:
   - `HOPSWORKS_API_KEY` (optional)
   - `HOPSWORKS_PROJECT_NAME` (optional)

#### Step 3: Deploy
- Space will auto-deploy on push to main branch
- Or manually trigger deployment from Space UI

#### Step 4: Access Your API
Once deployed, your API will be available at:
```
https://<your-username>-quakealert.hf.space
```

Test it:
```bash
curl https://<your-username>-quakealert.hf.space/
```

## CI/CD Setup

### GitHub Actions

The project includes `.github/workflows/ci-cd.yml` that:
1. Runs tests on every push
2. Builds Docker image
3. Deploys to Hugging Face (on main branch)

### Required Secrets

Add these in GitHub repository settings → Secrets:

1. **HF_TOKEN**: Hugging Face access token
   - Get from: https://huggingface.co/settings/tokens
   - Needs `write` permission

2. **HOPSWORKS_API_KEY** (optional): For feature store
3. **HOPSWORKS_PROJECT_NAME** (optional)

## Testing the Deployment

### Test API Endpoints

```bash
# Health check
curl https://your-space.hf.space/

# Analyze waveform
curl -X POST https://your-space.hf.space/analyze_waveform \
  -H "Content-Type: application/json" \
  -d '{
    "network": "IU",
    "station": "ANMO",
    "duration_minutes": 10
  }'

# Get stations
curl -X POST https://your-space.hf.space/stations \
  -H "Content-Type: application/json" \
  -d '{
    "network": "*",
    "station": "*"
  }'
```

### Test Dashboard

The dashboard is accessible at:
```
https://your-space.hf.space/dashboard
```

Or run locally:
```bash
streamlit run dashboard.py
```

## Troubleshooting

### Hugging Face Deployment Issues

1. **Build fails**: Check Dockerfile syntax
2. **Port issues**: Ensure app listens on port 7860 (HF default)
3. **Timeout**: Increase timeout in Space settings
4. **Memory**: Upgrade Space if needed (CPU/GPU)

### API Connection Issues

1. **CORS errors**: Already handled in `app/main.py`
2. **FDSN timeout**: IRIS service may be slow, increase timeout
3. **No data**: Check if station is online

### Local Development Issues

1. **ObsPy installation**: May need system libraries
   ```bash
   # On macOS
   brew install gcc
   
   # On Ubuntu
   sudo apt-get install gcc python3-dev
   ```

2. **Port already in use**: Change port in uvicorn command
   ```bash
   uvicorn app.main:app --port 8001
   ```

## Production Considerations

1. **Rate Limiting**: Add rate limiting for API endpoints
2. **Caching**: Cache station data to reduce FDSN calls
3. **Monitoring**: Add logging and monitoring
4. **Scaling**: Use multiple workers for FastAPI
   ```bash
   uvicorn app.main:app --workers 4
   ```

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `HOPSWORKS_API_KEY` | No | None | Hopsworks API key |
| `HOPSWORKS_PROJECT_NAME` | No | QuakeAlert | Hopsworks project name |
| `FDSN_CLIENT_URL` | No | IRIS | FDSN service URL |

## Next Steps

1. ✅ Deploy to Hugging Face Spaces
2. ✅ Set up CI/CD with GitHub Actions
3. ✅ Test all endpoints
4. ✅ Monitor performance
5. ✅ Add more monitoring stations
6. ✅ Implement model training pipeline (optional)

