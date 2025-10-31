FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies (exclude heavy transformers/torch for vLLM-only mode)
RUN pip install --no-cache-dir gradio openai python-dotenv requests

# Copy application files
COPY app.py .
COPY models/ ./models/
COPY data/ ./data/
COPY evaluation/ ./evaluation/

# Expose Gradio port
EXPOSE 7860

# Run the application
CMD ["python", "app.py"]
