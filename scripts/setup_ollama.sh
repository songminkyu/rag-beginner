#!/bin/bash

# EXAONE 모델 설정을 위한 Ollama 설정 스크립트
# Setup script for EXAONE models with Ollama

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
OLLAMA_HOST=${OLLAMA_HOST:-"http://localhost:11434"}
EXAONE_MODEL=${EXAONE_MODEL:-"exaone4:7.8b"}
INSTALL_OLLAMA=${INSTALL_OLLAMA:-"true"}

# Function to print colored output
print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Function to check if Ollama is installed
check_ollama_installed() {
    if command -v ollama &> /dev/null; then
        return 0
    else
        return 1
    fi
}

# Function to install Ollama
install_ollama() {
    print_info "Installing Ollama..."
    
    # Detect OS
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        # Linux
        curl -fsSL https://ollama.ai/install.sh | sh
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        if command -v brew &> /dev/null; then
            brew install ollama
        else
            print_warning "Homebrew not found. Please install Ollama manually from https://ollama.ai"
            return 1
        fi
    elif [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "win32" ]]; then
        # Windows
        print_warning "Please install Ollama manually from https://ollama.ai for Windows"
        print_info "After installation, restart your terminal and run this script again"
        return 1
    else
        print_error "Unsupported operating system: $OSTYPE"
        return 1
    fi
    
    print_success "Ollama installation completed"
}

# Function to start Ollama service
start_ollama_service() {
    print_info "Starting Ollama service..."
    
    # Check if Ollama is already running
    if curl -s "$OLLAMA_HOST/api/tags" &> /dev/null; then
        print_success "Ollama service is already running"
        return 0
    fi
    
    # Start Ollama in background
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        # Linux - use systemd if available
        if command -v systemctl &> /dev/null; then
            sudo systemctl start ollama || true
            sudo systemctl enable ollama || true
        else
            nohup ollama serve &> /dev/null &
        fi
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS - use launchd or background process
        if brew services list | grep -q ollama; then
            brew services start ollama
        else
            nohup ollama serve &> /dev/null &
        fi
    else
        # Other systems - start in background
        nohup ollama serve &> /dev/null &
    fi
    
    # Wait for service to start
    print_info "Waiting for Ollama service to start..."
    for i in {1..30}; do
        if curl -s "$OLLAMA_HOST/api/tags" &> /dev/null; then
            print_success "Ollama service is running"
            return 0
        fi
        sleep 2
    done
    
    print_error "Failed to start Ollama service"
    return 1
}

# Function to check available EXAONE models
check_available_models() {
    print_info "Checking available models..."
    
    # Get list of available models
    if ! MODELS=$(curl -s "$OLLAMA_HOST/api/tags" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    models = [model['name'] for model in data.get('models', [])]
    print('\n'.join(models))
except:
    pass
" 2>/dev/null); then
        print_warning "Could not retrieve model list"
        return 1
    fi
    
    if [ -n "$MODELS" ]; then
        print_info "Currently available models:"
        echo "$MODELS" | sed 's/^/  - /'
    else
        print_info "No models currently installed"
    fi
}

# Function to pull EXAONE model
pull_exaone_model() {
    local model_name=$1
    print_info "Downloading EXAONE model: $model_name"
    print_warning "This may take a while depending on your internet connection..."
    
    # Check if model is already available
    if curl -s "$OLLAMA_HOST/api/tags" | grep -q "$model_name"; then
        print_success "Model $model_name is already available"
        return 0
    fi
    
    # Download the model
    if ollama pull "$model_name"; then
        print_success "Successfully downloaded $model_name"
    else
        print_error "Failed to download $model_name"
        print_info "Available EXAONE models:"
        print_info "  - exaone4:2.4b   (2.4B parameters, ~1.3GB)"
        print_info "  - exaone4:7.8b   (7.8B parameters, ~4.1GB)"
        print_info "  - exaone4:32b    (32B parameters, ~18GB)"
        return 1
    fi
}

# Function to test EXAONE model
test_exaone_model() {
    local model_name=$1
    print_info "Testing EXAONE model: $model_name"
    
    # Korean test
    local korean_prompt="안녕하세요! 간단한 자기소개를 해주세요."
    print_info "Testing Korean prompt: $korean_prompt"
    
    if RESPONSE=$(ollama generate "$model_name" "$korean_prompt" 2>/dev/null); then
        print_success "Korean response:"
        echo "  $RESPONSE"
    else
        print_error "Failed to generate Korean response"
        return 1
    fi
    
    # English test
    local english_prompt="Hello! Please introduce yourself briefly."
    print_info "Testing English prompt: $english_prompt"
    
    if RESPONSE=$(ollama generate "$model_name" "$english_prompt" 2>/dev/null); then
        print_success "English response:"
        echo "  $RESPONSE"
    else
        print_error "Failed to generate English response"
        return 1
    fi
    
    print_success "Model testing completed successfully!"
}

# Function to create model configuration
create_model_config() {
    local model_name=$1
    local config_file="ollama_models.json"
    
    print_info "Creating model configuration: $config_file"
    
    cat > "$config_file" << EOF
{
  "exaone_models": {
    "exaone4:2.4b": {
      "size": "2.4B parameters",
      "memory_requirement": "4GB RAM",
      "description": "Smallest EXAONE model, suitable for testing and light workloads",
      "use_case": "Development, testing, resource-constrained environments"
    },
    "exaone4:7.8b": {
      "size": "7.8B parameters", 
      "memory_requirement": "8GB RAM",
      "description": "Balanced EXAONE model with good performance",
      "use_case": "General purpose, production workloads"
    },
    "exaone4:32b": {
      "size": "32B parameters",
      "memory_requirement": "32GB RAM", 
      "description": "Largest EXAONE model with best performance",
      "use_case": "High-performance requirements, research"
    }
  },
  "current_model": "$model_name",
  "ollama_host": "$OLLAMA_HOST",
  "setup_date": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
}
EOF
    
    print_success "Configuration saved to $config_file"
}

# Function to show usage guide
show_usage_guide() {
    print_info "🚀 EXAONE 사용 가이드 / Usage Guide"
    echo ""
    print_info "1. Python에서 사용하기 / Using in Python:"
    echo "  import ollama"
    echo "  response = ollama.generate('$EXAONE_MODEL', '안녕하세요!')"
    echo "  print(response['response'])"
    echo ""
    print_info "2. 명령줄에서 사용하기 / Using from command line:"
    echo "  ollama run $EXAONE_MODEL"
    echo ""
    print_info "3. API 엔드포인트 / API Endpoint:"
    echo "  curl -X POST $OLLAMA_HOST/api/generate \\"
    echo "    -H 'Content-Type: application/json' \\"
    echo "    -d '{"
    echo "      \"model\": \"$EXAONE_MODEL\","
    echo "      \"prompt\": \"안녕하세요! 간단한 인사를 해주세요.\","
    echo "      \"stream\": false"
    echo "    }'"
    echo ""
    print_info "4. 한국어 최적화 팁 / Korean Optimization Tips:"
    echo "  - EXAONE은 한국어에 특화되어 있습니다"
    echo "  - temperature를 0.1-0.6 사이로 설정하세요"
    echo "  - repeat_penalty는 1.0으로 설정하는 것을 권장합니다"
    echo ""
    print_info "5. 환경변수 설정 / Environment Variables:"
    echo "  export OLLAMA_BASE_URL=$OLLAMA_HOST"
    echo "  export EXAONE_MODEL_NAME=$EXAONE_MODEL"
}

# Function to show help
show_help() {
    echo "EXAONE Ollama Setup Script"
    echo ""
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -m, --model MODEL       EXAONE model to install (default: exaone4:7.8b)"
    echo "  -h, --host HOST         Ollama host URL (default: http://localhost:11434)"
    echo "  --no-install           Skip Ollama installation"
    echo "  --test-only            Only test existing model"
    echo "  --help                 Show this help message"
    echo ""
    echo "Available Models:"
    echo "  exaone4:2.4b          2.4B parameters (~1.3GB)"
    echo "  exaone4:7.8b          7.8B parameters (~4.1GB) [recommended]"
    echo "  exaone4:32b           32B parameters (~18GB)"
    echo ""
    echo "Examples:"
    echo "  $0                                    # Install with default settings"
    echo "  $0 -m exaone4:32b                   # Install 32B model"
    echo "  $0 --no-install --test-only         # Test existing installation"
}

# Main execution function
main() {
    print_info "🚀 EXAONE Ollama Setup Script"
    print_info "=============================="
    
    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            -m|--model)
                EXAONE_MODEL="$2"
                shift 2
                ;;
            -h|--host)
                OLLAMA_HOST="$2"
                shift 2
                ;;
            --no-install)
                INSTALL_OLLAMA="false"
                shift
                ;;
            --test-only)
                TEST_ONLY="true"
                shift
                ;;
            --help)
                show_help
                exit 0
                ;;
            *)
                print_error "Unknown option: $1"
                show_help
                exit 1
                ;;
        esac
    done
    
    print_info "Configuration:"
    print_info "  Model: $EXAONE_MODEL"  
    print_info "  Host: $OLLAMA_HOST"
    print_info "  Install Ollama: $INSTALL_OLLAMA"
    echo ""
    
    # Test-only mode
    if [[ "$TEST_ONLY" == "true" ]]; then
        print_info "Running in test-only mode"
        if check_ollama_installed; then
            check_available_models
            test_exaone_model "$EXAONE_MODEL"
            show_usage_guide
        else
            print_error "Ollama is not installed"
            exit 1
        fi
        exit 0
    fi
    
    # Install Ollama if needed
    if [[ "$INSTALL_OLLAMA" == "true" ]] && ! check_ollama_installed; then
        install_ollama
        if ! check_ollama_installed; then
            print_error "Ollama installation failed"
            exit 1
        fi
    elif check_ollama_installed; then
        print_success "Ollama is already installed"
    else
        print_error "Ollama is not installed. Use --no-install to skip installation."
        exit 1
    fi
    
    # Start Ollama service
    if ! start_ollama_service; then
        print_error "Failed to start Ollama service"
        exit 1
    fi
    
    # Check available models
    check_available_models
    
    # Pull EXAONE model
    if ! pull_exaone_model "$EXAONE_MODEL"; then
        print_error "Failed to download EXAONE model"
        exit 1
    fi
    
    # Test the model
    if ! test_exaone_model "$EXAONE_MODEL"; then
        print_error "Model testing failed"
        exit 1
    fi
    
    # Create configuration
    create_model_config "$EXAONE_MODEL"
    
    # Show usage guide
    show_usage_guide
    
    print_success "🎉 EXAONE setup completed successfully!"
    print_info "You can now use EXAONE with the RAG system."
}

# Run main function with all arguments
main "$@"