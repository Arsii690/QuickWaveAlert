# 🚀 QuakeAlertWave - Complete Deployment Guide

This guide walks you through setting up GitHub Actions CI/CD and Hugging Face deployment.

---

## 📋 Prerequisites

1. **GitHub Account** - For repository and CI/CD
2. **Hugging Face Account** - For deployment (free at huggingface.co)
3. **Docker Hub Account** (optional) - For container registry

---

## Step 1: Push to GitHub

### 1.1 Initialize Git Repository (if not done)

```bash
cd /Users/kainat/Desktop/QuakeAlertWave

# Initialize git (if not already)
git init

# Check .gitignore exists
cat .gitignore
```

### 1.2 Create GitHub Repository

1. Go to https://github.com/new
2. Repository name: `QuakeAlertWave`
3. Description: "Real-time seismic wave analysis with ML"
4. Choose **Public** or **Private**
5. **Don't** initialize with README (you already have one)
6. Click "Create repository"

### 1.3 Push Your Code

```bash
# Add all files
git add .

# Commit
git commit -m "Initial commit: QuakeAlertWave with CI/CD and deployment"

# Add remote (replace YOUR_USERNAME)
git remote add origin https://github.com/YOUR_USERNAME/QuakeAlertWave.git

# Push to main branch
git branch -M main
git push -u origin main
```

---

## Step 2: Set Up GitHub Secrets

### 2.1 Go to Repository Settings

1. Go to your GitHub repository
2. Click **Settings** → **Secrets and variables** → **Actions**
3. Click **New repository secret**

### 2.2 Add Required Secrets

#### For Hugging Face Deployment (REQUIRED):

| Secret Name | How to Get | Description |
|-------------|------------|-------------|
| `HF_TOKEN` | See below | Hugging Face API token |
| `HF_USERNAME` | Your HF username | Your Hugging Face username |
| `HF_SPACE_NAME` | Optional | Space name (default: QuakeAlertWave) |

**How to get HF_TOKEN:**
1. Go to https://huggingface.co/settings/tokens
2. Click **New token**
3. Name: `QuakeAlertWave-deploy`
4. Type: **Write** (for deployment)
5. Copy the token
6. Add as `HF_TOKEN` secret in GitHub

#### For Docker Hub (OPTIONAL):

| Secret Name | How to Get |
|-------------|------------|
| `DOCKER_USERNAME` | Your Docker Hub username |
| `DOCKER_PASSWORD` | Your Docker Hub password/token |

---

## Step 3: Create Hugging Face Space

### 3.1 Create New Space

1. Go to https://huggingface.co/new-space
2. **Space name:** `QuakeAlertWave` (or your choice)
3. **SDK:** Select **Docker**
4. **Hardware:** CPU Basic (free)
5. **Visibility:** Public
6. Click **Create Space**

### 3.2 Upload Files to Space

**Option A: Using Git (Recommended)**

```bash
# Install Hugging Face CLI
pip install huggingface_hub

# Login
huggingface-cli login
# Enter your HF_TOKEN when prompted

# Clone your space
git clone https://huggingface.co/spaces/YOUR_USERNAME/QuakeAlertWave
cd QuakeAlertWave

# Copy files from your project
cp /Users/kainat/Desktop/QuakeAlertWave/*.py .
cp /Users/kainat/Desktop/QuakeAlertWave/requirements.txt .
cp /Users/kainat/Desktop/QuakeAlertWave/Dockerfile.hf ./Dockerfile
cp /Users/kainat/Desktop/QuakeAlertWave/supervisord.conf .
cp /Users/kainat/Desktop/QuakeAlertWave/README_HF.md ./README.md
cp -r /Users/kainat/Desktop/QuakeAlertWave/app .

# Commit and push
git add .
git commit -m "Initial deployment"
git push
```

**Option B: Using Web Interface**

1. Go to your Space: https://huggingface.co/spaces/YOUR_USERNAME/QuakeAlertWave
2. Click **Files and versions** tab
3. Click **Add file** → **Upload files**
4. Upload these files:
   - `Dockerfile.hf` (rename to `Dockerfile`)
   - `supervisord.conf`
   - `requirements.txt`
   - `README_HF.md` (rename to `README.md`)
   - `dashboard.py`
   - `app/` folder (all files)

---

## Step 4: Verify CI/CD Pipeline

### 4.1 Check GitHub Actions

1. Go to your repository
2. Click **Actions** tab
3. You should see:
   - ✅ **test** job running
   - ✅ **build-docker** job (if on main branch)
   - ✅ **deploy-huggingface** job (if HF_TOKEN is set)

### 4.2 Fix Any Errors

If you see errors:
- Check the **Actions** tab for details
- Common issues:
  - Missing secrets → Add them in Settings
  - Import errors → Check requirements.txt
  - Docker build fails → Check Dockerfile

---

## Step 5: Test Deployment

### 5.1 Check Hugging Face Space

1. Go to: https://huggingface.co/spaces/YOUR_USERNAME/QuakeAlertWave
2. Click **App** tab
3. Wait for build to complete (5-10 minutes first time)
4. Your dashboard should be live!

### 5.2 Test Locally First

```bash
# Build Docker image
docker build -f Dockerfile.hf -t quakealert-test .

# Run container
docker run -p 7860:7860 quakealert-test

# Open http://localhost:7860
```

---

## 📊 Deployment Status Checklist

| Task | Status | Notes |
|------|--------|-------|
| Code pushed to GitHub | ⬜ | Step 1 |
| GitHub Secrets added | ⬜ | Step 2 |
| HF Space created | ⬜ | Step 3 |
| Files uploaded to HF | ⬜ | Step 3 |
| CI/CD pipeline passing | ⬜ | Step 4 |
| HF Space building | ⬜ | Step 5 |
| Dashboard live | ⬜ | Step 5 |

---

## 🔧 Troubleshooting

### CI/CD Not Running?

1. Check branch name (must be `main` or `master`)
2. Check if workflow file is in `.github/workflows/`
3. Check Actions tab for errors

### HF Deployment Fails?

1. Verify `HF_TOKEN` has **Write** permissions
2. Check `HF_USERNAME` is correct
3. Verify `Dockerfile` exists in Space (not `Dockerfile.hf`)

### Docker Build Fails?

1. Check `requirements.txt` for all dependencies
2. Verify Python version (3.11)
3. Check Dockerfile syntax

---

## 🎯 Quick Commands Reference

```bash
# Push to GitHub
git add .
git commit -m "Update"
git push origin main

# Deploy to HF manually
huggingface-cli login
huggingface-cli upload YOUR_USERNAME/QuakeAlertWave \
  --repo-type space \
  . \
  --exclude "venv/**" "*.pyc"

# Check CI/CD status
# Go to: https://github.com/YOUR_USERNAME/QuakeAlertWave/actions
```

---

## ✅ Success Criteria

Your project is fully deployed when:

1. ✅ GitHub Actions shows green checkmarks
2. ✅ Hugging Face Space builds successfully
3. ✅ Dashboard loads at: https://huggingface.co/spaces/YOUR_USERNAME/QuakeAlertWave
4. ✅ You can analyze waveforms in the live dashboard

---

**Need help?** Check the logs in:
- GitHub Actions → Your workflow run
- Hugging Face Space → Logs tab

