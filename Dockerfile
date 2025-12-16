# 1. Base Image (Python 3.11)
FROM python:3.11-slim

# 2. Safety Setting: Force Python to print logs immediately
ENV PYTHONUNBUFFERED=1

# 3. Set the folder inside the container
WORKDIR /app

# 4. Install System Dependencies
# Required for ObsPy and scientific computing libraries
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    python3-dev \
    libffi-dev \
    libssl-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 5. Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 6. Copy the rest of your application code
COPY . .

# 7. Open ports
# 8000 for FastAPI, 8501 for Streamlit (if needed)
EXPOSE 8000 8501

# 8. Start the API Server
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

