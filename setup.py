from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="llm-rag-learning",
    version="1.0.0",
    author="LLM RAG Learner",
    author_email="learner@example.com",
    description="2025년 최신 LLM RAG 기술 학습을 위한 통합 저장소",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/llm-rag-learning",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=8.3.0",
            "black>=24.8.0",
            "flake8>=7.1.0",
            "mypy>=1.11.0",
            "pre-commit>=3.8.0",
        ],
        "monitoring": [
            "wandb>=0.17.0",
            "tensorboard>=2.17.0",
        ],
        "korean": [
            "konlpy>=0.6.0",
            "kss>=4.5.4",
        ],
        "gpu": [
            "torch>=2.3.0+cu121",
            "faiss-gpu>=1.8.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "rag-setup=scripts.setup_ollama:main",
            "rag-eval=scripts.run_evaluation:main",
            "rag-serve=projects.api_server:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.md", "*.txt", "*.yaml", "*.yml", "*.json"],
        "data": ["documents/**/*", "datasets/**/*"],
        "configs": ["*.yaml", "*.yml", "*.json"],
    },
    zip_safe=False,
)