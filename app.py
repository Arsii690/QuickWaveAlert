"""
Hugging Face Spaces Entry Point
This file is used when deploying to Hugging Face Spaces.
It wraps the FastAPI app for Space deployment.
"""

import uvicorn
from app.main import app

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=7860,  # Hugging Face Spaces default port
        log_level="info"
    )

