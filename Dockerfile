# Malaysian English Transcription POC - Docker Configuration
# ========================================================

FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    make \
    libasound2-dev \
    portaudio19-dev \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY complete_malaysian_transcription_poc.py .
COPY config.ini .
COPY test_cases.py .

# Create data directories
RUN mkdir -p data/audio_uploads data/sample_audio logs

# Expose Streamlit port
EXPOSE 8501

# Health check
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# Run the application
CMD ["streamlit", "run", "complete_malaysian_transcription_poc.py", "--server.port=8501", "--server.address=0.0.0.0"]