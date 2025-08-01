# Multi-stage build for RAG Beginner project
FROM python:3.11-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Development stage
FROM base as development

# Install development dependencies
RUN pip install \
    jupyter \
    ipython \
    pytest \
    pytest-cov \
    black \
    isort \
    flake8 \
    mypy

# Copy source code
COPY . .

# Install the package in development mode
RUN pip install -e .

# Create data directories
RUN mkdir -p /app/data/vector_stores /app/data/documents /app/logs

# Expose ports for Jupyter and API server
EXPOSE 8888 8000 7860

# Default command for development
CMD ["python", "-m", "src.tutorials.01_getting_started.hello_rag", "--mode", "demo"]

# Production stage
FROM base as production

# Copy only necessary files
COPY src/ ./src/
COPY setup.py pyproject.toml ./
COPY docker/entrypoint.sh ./entrypoint.sh

# Install the package
RUN pip install .

# Create non-root user
RUN useradd --create-home --shell /bin/bash appuser && \
    chown -R appuser:appuser /app

# Create data directories with proper permissions
RUN mkdir -p /app/data/vector_stores /app/data/documents /app/logs && \
    chown -R appuser:appuser /app/data /app/logs

# Switch to non-root user
USER appuser

# Make entrypoint script executable
RUN chmod +x ./entrypoint.sh

# Expose API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command
ENTRYPOINT ["./entrypoint.sh"]
CMD ["api"]

# Gradio demo stage  
FROM base as gradio

# Copy source code
COPY src/ ./src/
COPY setup.py pyproject.toml ./

# Install the package
RUN pip install .

# Install Gradio
RUN pip install gradio

# Create data directories
RUN mkdir -p /app/data/vector_stores /app/data/documents

# Expose Gradio port
EXPOSE 7860

# Run Gradio interface
CMD ["python", "-c", "import gradio as gr; import sys; sys.path.append('./src'); from projects.chatbot.app import demo; demo.launch(server_name='0.0.0.0', server_port=7860)"]

# Jupyter stage for development and experimentation
FROM development as jupyter

# Set Jupyter configuration
RUN mkdir -p /home/root/.jupyter && \
    echo "c.NotebookApp.ip = '0.0.0.0'" >> /home/root/.jupyter/jupyter_notebook_config.py && \
    echo "c.NotebookApp.port = 8888" >> /home/root/.jupyter/jupyter_notebook_config.py && \
    echo "c.NotebookApp.open_browser = False" >> /home/root/.jupyter/jupyter_notebook_config.py && \
    echo "c.NotebookApp.allow_root = True" >> /home/root/.jupyter/jupyter_notebook_config.py

# Expose Jupyter port
EXPOSE 8888

# Start Jupyter Lab
CMD ["jupyter", "lab", "--allow-root", "--no-browser", "--ip=0.0.0.0", "--port=8888"]