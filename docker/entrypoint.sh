#!/bin/bash

# Entrypoint script for RAG Beginner Docker container

set -e

# Default values
COMMAND=${1:-api}
LOG_LEVEL=${LOG_LEVEL:-INFO}

# Function to wait for service
wait_for_service() {
    local host=$1
    local port=$2
    local service_name=$3
    
    echo "Waiting for $service_name to be ready..."
    while ! nc -z $host $port; do
        sleep 1
    done
    echo "$service_name is ready!"
}

# Function to setup environment
setup_environment() {
    echo "Setting up environment..."
    
    # Create necessary directories
    mkdir -p /app/data/vector_stores
    mkdir -p /app/data/documents  
    mkdir -p /app/logs
    
    # Set proper permissions
    chmod 755 /app/data
    chmod 755 /app/logs
    
    echo "Environment setup complete!"
}

# Function to check dependencies
check_dependencies() {
    echo "Checking dependencies..."
    
    # Check if ChromaDB is available (if configured)
    if [ ! -z "$CHROMADB_HOST" ]; then
        wait_for_service ${CHROMADB_HOST:-chroma} ${CHROMADB_PORT:-8000} "ChromaDB"
    fi
    
    # Check if Redis is available (if configured)
    if [ ! -z "$REDIS_HOST" ]; then
        wait_for_service ${REDIS_HOST:-redis} ${REDIS_PORT:-6379} "Redis"
    fi
    
    echo "Dependencies check complete!"
}

# Function to run database migrations or setup
run_setup() {
    echo "Running setup tasks..."
    
    # Initialize vector stores if needed
    python -c "
import os
import sys
sys.path.append('/app')

try:
    # Test basic imports
    from src.core.llm_providers.base_provider import BaseLLMProvider
    print('✅ Core modules imported successfully')
    
    # Test vector store connection
    if os.getenv('CHROMADB_HOST'):
        print('✅ ChromaDB configuration detected')
        
    print('✅ Setup tasks completed')
except Exception as e:
    print(f'❌ Setup failed: {e}')
    sys.exit(1)
"
}

# Function to start API server
start_api() {
    echo "Starting RAG API server..."
    
    export PYTHONPATH=/app
    
    # Check if we have an API server implementation
    if [ -f "/app/src/projects/api_server.py" ]; then
        exec python -m src.projects.api_server
    else
        # Fallback to basic server
        exec python -c "
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse

app = FastAPI(title='RAG Beginner API', version='0.1.0')

@app.get('/health')
async def health_check():
    return JSONResponse({'status': 'healthy', 'service': 'rag-api'})

@app.get('/')
async def root():
    return JSONResponse({'message': 'RAG Beginner API', 'version': '0.1.0'})

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)
"
    fi
}

# Function to start demo
start_demo() {
    echo "Starting RAG demo..."
    
    export PYTHONPATH=/app
    exec python -m src.tutorials.01_getting_started.hello_rag --mode demo
}

# Function to start interactive mode
start_interactive() {
    echo "Starting RAG interactive mode..."
    
    export PYTHONPATH=/app
    exec python -m src.tutorials.01_getting_started.hello_rag --mode interactive
}

# Function to start Jupyter
start_jupyter() {
    echo "Starting Jupyter Lab..."
    
    export PYTHONPATH=/app
    exec jupyter lab \
        --allow-root \
        --no-browser \
        --ip=0.0.0.0 \
        --port=8888 \
        --ServerApp.token='' \
        --ServerApp.password='' \
        --ServerApp.allow_origin='*' \
        --ServerApp.base_url=${JUPYTER_BASE_URL:-/}
}

# Function to run tests
run_tests() {
    echo "Running tests..."
    
    export PYTHONPATH=/app
    
    if [ -d "/app/tests" ]; then
        exec python -m pytest /app/tests -v
    else
        echo "No tests directory found"
        exec python -c "
import sys
sys.path.append('/app')

# Basic import test
try:
    from src.core.llm_providers.base_provider import BaseLLMProvider
    print('✅ Basic imports successful')
    
    from src.tutorials.01_getting_started.hello_rag import SimpleRAG
    print('✅ HelloRAG import successful')
    
    print('✅ All basic tests passed')
except Exception as e:
    print(f'❌ Test failed: {e}')
    sys.exit(1)
"
    fi
}

# Function to show help
show_help() {
    echo "RAG Beginner Docker Container"
    echo ""
    echo "Usage: docker run rag-beginner [COMMAND]"
    echo ""
    echo "Commands:"
    echo "  api          Start the API server (default)"
    echo "  demo         Run the RAG demo"
    echo "  interactive  Start interactive RAG mode"
    echo "  jupyter      Start Jupyter Lab"
    echo "  test         Run tests"
    echo "  bash         Start bash shell"
    echo "  help         Show this help message"
    echo ""
    echo "Environment Variables:"
    echo "  LOG_LEVEL              Logging level (DEBUG, INFO, WARNING, ERROR)"
    echo "  OPENAI_API_KEY         OpenAI API key"
    echo "  ANTHROPIC_API_KEY      Anthropic API key"
    echo "  CHROMADB_HOST          ChromaDB host (default: chroma)"
    echo "  CHROMADB_PORT          ChromaDB port (default: 8000)"
    echo "  REDIS_HOST             Redis host (optional)"
    echo "  REDIS_PORT             Redis port (default: 6379)"
}

# Main execution
main() {
    echo "RAG Beginner Container Starting..."
    echo "Command: $COMMAND"
    echo "Log Level: $LOG_LEVEL"
    echo ""
    
    # Setup environment
    setup_environment
    
    case $COMMAND in
        api)
            check_dependencies
            run_setup
            start_api
            ;;
        demo)
            check_dependencies
            run_setup
            start_demo
            ;;
        interactive)
            check_dependencies
            run_setup
            start_interactive
            ;;
        jupyter)
            start_jupyter
            ;;
        test)
            run_tests
            ;;
        bash)
            exec /bin/bash
            ;;
        help|--help|-h)
            show_help
            ;;
        *)
            echo "Unknown command: $COMMAND"
            show_help
            exit 1
            ;;
    esac
}

# Install netcat if not available (for service waiting)
if ! command -v nc &> /dev/null; then
    echo "Installing netcat..."
    apt-get update && apt-get install -y netcat-openbsd
fi

# Run main function
main "$@"