#!/usr/bin/env python3
"""
EXAONE 4.0 모델 설정 스크립트
Hugging Face Transformers를 통한 EXAONE 4.0 모델 설정
"""

import os
import sys
import subprocess
import logging
import torch
from pathlib import Path
from typing import Optional, Dict, Any

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 색상 출력을 위한 ANSI 코드
class Colors:
    RED = '\033[0;31m'
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    BLUE = '\033[0;34m'
    PURPLE = '\033[0;35m'
    CYAN = '\033[0;36m'
    WHITE = '\033[1;37m'
    NC = '\033[0m'  # No Color

def print_colored(message: str, color: str = Colors.WHITE):
    """색상 출력"""
    print(f"{color}{message}{Colors.NC}")

def print_info(message: str):
    print_colored(f"ℹ️  {message}", Colors.BLUE)

def print_success(message: str):
    print_colored(f"✅ {message}", Colors.GREEN)

def print_warning(message: str):
    print_colored(f"⚠️  {message}", Colors.YELLOW)

def print_error(message: str):
    print_colored(f"❌ {message}", Colors.RED)

def check_system_requirements():
    """시스템 요구사항 확인"""
    print_info("시스템 요구사항 확인 중...")
    
    # Python 버전 확인
    python_version = sys.version_info
    if python_version < (3, 9):
        print_error(f"Python 3.9 이상이 필요합니다. 현재: {python_version.major}.{python_version.minor}")
        return False
    else:
        print_success(f"Python 버전: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    # PyTorch 설치 확인
    try:
        import torch
        print_success(f"PyTorch 버전: {torch.__version__}")
        
        # CUDA 사용 가능 여부 확인
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print_success(f"CUDA 사용 가능: {gpu_count}개 GPU, {gpu_name}, {memory:.1f}GB")
        else:
            print_warning("CUDA를 사용할 수 없습니다. CPU 모드로 실행됩니다.")
            
    except ImportError:
        print_error("PyTorch가 설치되어 있지 않습니다.")
        return False
    
    # 메모리 확인
    try:
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            if gpu_memory < 16:
                print_warning(f"GPU 메모리가 부족할 수 있습니다: {gpu_memory:.1f}GB")
                print_info("EXAONE-4.0-32B 모델은 최소 24GB GPU 메모리를 권장합니다.")
            else:
                print_success(f"충분한 GPU 메모리: {gpu_memory:.1f}GB")
        
        # RAM 확인 (Linux/macOS)
        if sys.platform != "win32":
            import psutil
            ram_gb = psutil.virtual_memory().total / 1024**3
            print_info(f"시스템 RAM: {ram_gb:.1f}GB")
            
    except Exception as e:
        print_warning(f"메모리 정보를 확인할 수 없습니다: {e}")
    
    return True

def install_dependencies():
    """필요한 의존성 설치"""
    print_info("EXAONE 지원을 위한 의존성 설치 중...")
    
    # 필수 패키지 목록
    required_packages = [
        "torch>=2.3.0",
        "accelerate>=0.32.0",
        "sentence-transformers>=3.0.0",
        "bitsandbytes",  # 메모리 최적화
        "flash-attn",    # 주의: 설치 실패할 수 있음
    ]
    
    # EXAONE 지원 transformers 설치
    print_info("EXAONE 지원 transformers 설치 중...")
    try:
        subprocess.run([
            sys.executable, "-m", "pip", "install", 
            "git+https://github.com/lgai-exaone/transformers@add-exaone4"
        ], check=True, capture_output=True)
        print_success("EXAONE 지원 transformers 설치 완료")
    except subprocess.CalledProcessError as e:
        print_warning("EXAONE 지원 transformers 설치 실패, 기본 transformers 사용")
        subprocess.run([sys.executable, "-m", "pip", "install", "transformers>=4.45.0"], check=True)
    
    # 기타 패키지 설치
    for package in required_packages:
        try:
            print_info(f"설치 중: {package}")
            subprocess.run([sys.executable, "-m", "pip", "install", package], 
                         check=True, capture_output=True)
            print_success(f"✓ {package}")
        except subprocess.CalledProcessError:
            if "flash-attn" in package:
                print_warning(f"✗ {package} (선택사항, 설치 실패)")
            else:
                print_error(f"✗ {package} 설치 실패")

def get_model_recommendations():
    """시스템에 맞는 모델 추천"""
    recommendations = {}
    
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        
        if gpu_memory >= 32:
            recommendations["recommended"] = "LGAI-EXAONE/EXAONE-4.0-32B"
            recommendations["reason"] = "충분한 GPU 메모리로 최고 성능 모델 사용 가능"
        elif gpu_memory >= 8:
            recommendations["recommended"] = "LGAI-EXAONE/EXAONE-4.0-1.2B"
            recommendations["reason"] = "GPU 메모리에 맞는 소형 모델 권장"
        else:
            recommendations["recommended"] = "LGAI-EXAONE/EXAONE-4.0-1.2B"
            recommendations["reason"] = "제한된 GPU 메모리로 소형 모델 필수"
    else:
        recommendations["recommended"] = "LGAI-EXAONE/EXAONE-4.0-1.2B"
        recommendations["reason"] = "CPU 환경에서는 소형 모델 권장"
    
    return recommendations

def test_model_loading(model_name: str) -> bool:
    """모델 로딩 테스트"""
    print_info(f"모델 로딩 테스트: {model_name}")
    
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        # 토크나이저 로드 테스트
        print_info("토크나이저 로딩 중...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        print_success("토크나이저 로딩 성공")
        
        # 모델 로드 테스트 (메타데이터만)
        print_info("모델 메타데이터 확인 중...")
        model_config = AutoModelForCausalLM.from_pretrained(
            model_name, 
            torch_dtype=torch.bfloat16,
            device_map="cpu",  # 메타데이터만 확인
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )
        print_success("모델 메타데이터 확인 성공")
        
        # 간단한 토크나이징 테스트
        test_text = "안녕하세요, EXAONE 모델 테스트입니다."
        tokens = tokenizer.encode(test_text)
        decoded = tokenizer.decode(tokens)
        
        print_success(f"토크나이징 테스트 성공: {len(tokens)} 토큰")
        
        return True
        
    except Exception as e:
        print_error(f"모델 테스트 실패: {e}")
        return False

def create_test_script():
    """테스트 스크립트 생성"""
    test_script_content = '''#!/usr/bin/env python3
"""
EXAONE 4.0 모델 테스트 스크립트
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_exaone_model(model_name="LGAI-EXAONE/EXAONE-4.0-1.2B"):
    """EXAONE 모델 테스트"""
    
    print(f"🚀 EXAONE 모델 테스트: {model_name}")
    print("=" * 50)
    
    try:
        # 토크나이저 로드
        logger.info("토크나이저 로딩 중...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # 모델 로드
        logger.info("모델 로딩 중...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )
        
        print(f"✅ 모델 로딩 성공!")
        print(f"📍 디바이스: {model.device}")
        print(f"🧮 dtype: {model.dtype}")
        
        if torch.cuda.is_available():
            memory_used = torch.cuda.memory_allocated() / 1024**3
            print(f"💾 GPU 메모리 사용량: {memory_used:.2f} GB")
        
        # 테스트 생성
        test_prompts = [
            "안녕하세요! 간단한 자기소개를 해주세요.",
            "인공지능의 미래에 대해 어떻게 생각하시나요?",
            "한국의 전통 음식 중 하나를 추천해주세요."
        ]
        
        for i, prompt in enumerate(test_prompts, 1):
            print(f"\\n🔄 테스트 {i}: {prompt}")
            print("-" * 30)
            
            messages = [{"role": "user", "content": prompt}]
            input_ids = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt"
            )
            
            with torch.no_grad():
                output = model.generate(
                    input_ids.to(model.device),
                    max_new_tokens=128,
                    temperature=0.7,
                    do_sample=True,
                    top_p=0.95,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            generated_tokens = output[0][len(input_ids[0]):]
            response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            
            print(f"💬 응답: {response.strip()}")
        
        print("\\n🎉 모든 테스트 통과!")
        
    except Exception as e:
        print(f"❌ 테스트 실패: {e}")
        return False
    
    return True

if __name__ == "__main__":
    import sys
    
    model_name = sys.argv[1] if len(sys.argv) > 1 else "LGAI-EXAONE/EXAONE-4.0-1.2B"
    test_exaone_model(model_name)
'''
    
    # 테스트 스크립트 파일 생성
    test_script_path = Path("scripts/test_exaone.py")
    test_script_path.parent.mkdir(exist_ok=True)
    
    with open(test_script_path, "w", encoding="utf-8") as f:
        f.write(test_script_content)
    
    # 실행 권한 부여 (Unix 계열)
    if sys.platform != "win32":
        import stat
        st = os.stat(test_script_path)
        os.chmod(test_script_path, st.st_mode | stat.S_IEXEC)
    
    print_success(f"테스트 스크립트 생성: {test_script_path}")
    return test_script_path

def update_env_file():
    """환경변수 파일 업데이트"""
    print_info("환경변수 설정 업데이트 중...")
    
    env_file = Path(".env")
    env_example = Path(".env.example")
    
    # .env 파일이 없으면 .env.example에서 복사
    if not env_file.exists() and env_example.exists():
        import shutil
        shutil.copy(env_example, env_file)
        print_success(".env 파일 생성 완료")
    
    # 권장 모델 설정
    recommendations = get_model_recommendations()
    recommended_model = recommendations["recommended"]
    
    print_info(f"권장 모델: {recommended_model}")
    print_info(f"이유: {recommendations['reason']}")
    
    # .env 파일 업데이트
    if env_file.exists():
        with open(env_file, "r", encoding="utf-8") as f:
            content = f.read()
        
        # EXAONE_MODEL 설정 업데이트
        if "EXAONE_MODEL=" in content:
            # 기존 설정 주석 처리하고 새로운 설정 추가
            lines = content.split("\n")
            new_lines = []
            
            for line in lines:
                if line.startswith("EXAONE_MODEL="):
                    new_lines.append(f"# {line}")
                    new_lines.append(f"EXAONE_MODEL={recommended_model}")
                else:
                    new_lines.append(line)
            
            content = "\n".join(new_lines)
        else:
            # 새로운 설정 추가
            content += f"\n# EXAONE 4.0 Model Configuration\nEXAONE_MODEL={recommended_model}\n"
        
        with open(env_file, "w", encoding="utf-8") as f:
            f.write(content)
        
        print_success("환경변수 파일 업데이트 완료")

def main():
    """메인 함수"""
    print_colored("🚀 EXAONE 4.0 모델 설정 스크립트", Colors.CYAN)
    print_colored("=" * 50, Colors.CYAN)
    print()
    
    # 1. 시스템 요구사항 확인
    if not check_system_requirements():
        print_error("시스템 요구사항을 만족하지 않습니다.")
        return False
    
    print()
    
    # 2. 의존성 설치
    try:
        install_dependencies()
        print_success("의존성 설치 완료")
    except Exception as e:
        print_error(f"의존성 설치 실패: {e}")
        return False
    
    print()
    
    # 3. 모델 추천 및 테스트
    recommendations = get_model_recommendations()
    recommended_model = recommendations["recommended"]
    
    print_info(f"권장 모델: {recommended_model}")
    print_info(f"이유: {recommendations['reason']}")
    
    # 사용자 선택
    print("\n사용할 모델을 선택하세요:")
    print("1) LGAI-EXAONE/EXAONE-4.0-1.2B (소형, 빠름)")
    print("2) LGAI-EXAONE/EXAONE-4.0-32B (대형, 고성능)")
    print(f"3) 권장 모델 사용 ({recommended_model})")
    
    choice = input("\n선택 (1-3): ").strip()
    
    model_map = {
        "1": "LGAI-EXAONE/EXAONE-4.0-1.2B",
        "2": "LGAI-EXAONE/EXAONE-4.0-32B", 
        "3": recommended_model
    }
    
    selected_model = model_map.get(choice, recommended_model)
    
    print_info(f"선택된 모델: {selected_model}")
    
    # 4. 모델 테스트
    if test_model_loading(selected_model):
        print_success("모델 로딩 테스트 성공!")
    else:
        print_warning("모델 로딩 테스트 실패. 설정을 확인해주세요.")
    
    print()
    
    # 5. 환경변수 업데이트
    update_env_file()
    
    # 6. 테스트 스크립트 생성
    test_script_path = create_test_script()
    
    # 7. 사용법 안내
    print()
    print_colored("🎉 EXAONE 4.0 설정 완료!", Colors.GREEN)
    print("=" * 50)
    print()
    print("다음 단계:")
    print(f"1. 전체 테스트: python {test_script_path}")
    print("2. Hello RAG 실행: python tutorials/01_getting_started/hello_rag.py")
    print("3. 모델 변경: .env 파일에서 EXAONE_MODEL 수정")
    print()
    print("문제 해결:")
    print("- GPU 메모리 부족: 더 작은 모델 사용 또는 batch_size 감소")
    print("- 속도 개선: GPU 사용, flash-attention 설치")
    print("- 한국어 성능: EXAONE 모델이 한국어에 최적화되어 있음")
    print()
    print_success("즐거운 RAG 학습 되세요! 🚀")
    
    return True

if __name__ == "__main__":
    main()