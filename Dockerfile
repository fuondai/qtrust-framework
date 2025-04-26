FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install PyTorch CPU version
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Copy QTrust code
COPY . .

# Create necessary directories
RUN mkdir -p benchmark_results/{logs,data,charts} demo_results

# Set environment variables
ENV PYTHONPATH=/app

# Default command
CMD ["python", "qtrust_main.py", "--log-level", "INFO"]

# Usage instructions in the form of a label
LABEL usage="To run the demo: docker run qtrust\n\
    To run the benchmark: docker run qtrust python qtrust_main.py --benchmark all\n\
    To generate visualizations: docker run qtrust python scripts/generate_visuals.py"
