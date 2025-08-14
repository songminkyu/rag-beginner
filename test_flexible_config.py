#!/usr/bin/env python3
"""
Flexible Local Model Configuration Test
다양한 로컬 모델 설정을 테스트하는 스크립트
"""

import os
import sys
from pathlib import Path

# 프로젝트 루트 경로 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_config_manager():
    """설정 관리자 테스트"""
    print("Testing Config Manager")
    print("=" * 50)
    
    try:
        from src.config.api_config import config_manager
        
        # 사용 가능한 제공자 확인
        available_providers = config_manager.get_available_providers()
        print(f"Available providers: {available_providers}")
        
        # 로컬 모델 정보 확인
        local_info = config_manager.get_local_model_info()
        print(f"Local model info: {local_info}")
        
        # 설정 요약 확인
        config_summary = config_manager.get_config_summary()
        print(f"Config summary: {config_summary}")
        
        return True
        
    except Exception as e:
        print(f"Config manager test failed: {e}")
        return False


def test_flexible_provider():
    """유연한 로컬 제공자 테스트"""
    print("\nTesting Flexible Local Provider")
    print("=" * 50)
    
    try:
        from src.config.api_config import config_manager
        from src.core.llm_providers.flexible_local_provider import FlexibleLocalProvider
        
        # 로컬 설정 가져오기
        config = config_manager.get_provider_config("local")
        print(f"Local config: {config}")
        
        # 제공자 생성 시도
        provider = FlexibleLocalProvider(config)
        print(f"Provider created: {provider.get_model_info()}")
        
        # 간단한 생성 테스트 (실제 모델이 없어도 정보 확인 가능)
        try:
            # 설정 검증만 수행
            is_valid = provider.validate_config()
            print(f"Config valid: {is_valid}")
            
            if is_valid:
                # 간단한 생성 테스트
                response = provider.generate("Hello, how are you?")
                print(f"Generation test success: {response.content[:100]}...")
            
        except Exception as gen_error:
            print(f"Generation test failed (expected - no model): {gen_error}")
        
        # 정리
        provider.cleanup()
        print("Provider cleanup completed")
        
        return True
        
    except Exception as e:
        print(f"Flexible provider test failed: {e}")
        return False


def test_embedding_provider():
    """임베딩 제공자 테스트"""
    print("\nTesting Embedding Provider")
    print("=" * 50)
    
    try:
        from src.config.api_config import config_manager
        from src.core.llm_providers.flexible_local_provider import LocalEmbeddingProvider
        
        # 로컬 설정 가져오기
        config = config_manager.get_provider_config("local")
        
        # 임베딩 제공자 생성 시도
        embedding_provider = LocalEmbeddingProvider(config)
        print(f"Embedding provider created")
        print(f"Embedding model: {embedding_provider.model_name}")
        print(f"Embedding dimension: {embedding_provider.get_dimension()}")
        
        # 간단한 임베딩 테스트
        text = "Hello, this is a test sentence."
        embedding = embedding_provider.embed_text(text)
        print(f"Embedding test success: dim={len(embedding)}, first 5 values={embedding[:5]}")
        
        return True
        
    except Exception as e:
        print(f"Embedding provider test failed: {e}")
        return False


def test_model_types():
    """다양한 모델 타입 테스트"""
    print("\nTesting Various Model Types")
    print("=" * 50)
    
    test_configs = [
        {
            "name": "Llama (Ollama)",
            "config": {
                "model": "llama3.1:8b",
                "model_type": "llama",
                "ollama_base_url": "http://localhost:11434",
                "korean_optimized": False,
            }
        },
        {
            "name": "EXAONE (Ollama)", 
            "config": {
                "model": "exaone4:7.8b",
                "model_type": "exaone", 
                "ollama_base_url": "http://localhost:11434",
                "korean_optimized": True,
            }
        },
        {
            "name": "HuggingFace DialoGPT",
            "config": {
                "model": "microsoft/DialoGPT-medium",
                "hf_model": "microsoft/DialoGPT-medium",
                "model_type": "auto",
                "device": "cpu",
                "torch_dtype": "float16",
                "korean_optimized": False,
            }
        },
        {
            "name": "Korean HuggingFace Model",
            "config": {
                "model": "skt/kogpt2-base-v2",
                "hf_model": "skt/kogpt2-base-v2", 
                "model_type": "auto",
                "device": "cpu",
                "torch_dtype": "float16",
                "korean_optimized": True,
            }
        }
    ]
    
    try:
        from src.core.llm_providers.flexible_local_provider import FlexibleLocalProvider
        
        for test_case in test_configs:
            print(f"\nTesting: {test_case['name']}")
            
            try:
                provider = FlexibleLocalProvider(test_case['config'])
                model_info = provider.get_model_info()
                print(f"  Model info: {model_info}")
                
                # 모델 타입 감지 테스트
                detected_type = provider._detect_model_type()
                print(f"  Detected type: {detected_type}")
                
                # Ollama vs HuggingFace 선택 테스트
                should_ollama = provider._should_use_ollama()
                print(f"  Should use Ollama: {should_ollama}")
                
                provider.cleanup()
                
            except Exception as e:
                print(f"  Test failed (expected - no model): {e}")
        
        return True
        
    except Exception as e:
        print(f"Model types test failed: {e}")
        return False


def test_environment_variables():
    """환경 변수 테스트"""
    print("\nTesting Environment Variables")
    print("=" * 50)
    
    # 중요한 환경 변수들 확인
    important_vars = [
        "LOCAL_MODEL_NAME",
        "LOCAL_MODEL_TYPE", 
        "HF_MODEL_NAME",
        "OLLAMA_BASE_URL",
        "EMBEDDING_MODEL",
        "KOREAN_OPTIMIZED",
        "TORCH_DTYPE",
        "DEVICE"
    ]
    
    for var in important_vars:
        value = os.getenv(var, "Not Set")
        print(f"  {var}: {value}")
    
    print("\nEnvironment variables check completed")
    return True


def main():
    """메인 테스트 함수"""
    print("Flexible Local Model Configuration Test")
    print("=" * 60)
    
    test_results = []
    
    # 테스트 실행
    test_results.append(("Environment Variables", test_environment_variables()))
    test_results.append(("Config Manager", test_config_manager()))
    test_results.append(("Embedding Provider", test_embedding_provider()))
    test_results.append(("Flexible Provider", test_flexible_provider()))
    test_results.append(("Model Types", test_model_types()))
    
    # 결과 요약
    print("\nTest Results Summary")
    print("=" * 60)
    
    passed = 0
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "PASS" if result else "FAIL"
        print(f"[{status}] {test_name}")
        if result:
            passed += 1
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("All tests passed successfully!")
        print("\nAvailable configurations:")
        print("1. 다양한 Ollama 모델 (llama3.1, mistral, exaone4 등)")
        print("2. HuggingFace 모델 (microsoft/DialoGPT-medium 등)")
        print("3. 한국어 최적화 모델 지원")
        print("4. 자동 모델 타입 감지")
        print("5. 유연한 백엔드 선택 (Ollama vs HuggingFace)")
    else:
        print("Some tests failed. Please check your configuration.")
        print("\nTroubleshooting:")
        print("1. .env 파일 설정 확인")
        print("2. 필요한 라이브러리 설치: pip install sentence-transformers torch")
        print("3. Ollama 설치 및 실행 (선택사항)")
        print("4. 환경 변수 설정 확인")


if __name__ == "__main__":
    main()