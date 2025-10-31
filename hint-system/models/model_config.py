"""
로컬 환경에서 테스트할 가벼운 모델 설정

vLLM 지원 모델:
- vLLM은 HuggingFace Transformers 대비 15-24배 빠른 추론 속도 제공
- Continuous batching과 PagedAttention으로 메모리 효율성 극대화
- 4GB+ VRAM GPU에서 7B 모델, 24GB+ VRAM에서 13B-34B 모델 권장
"""

# ============================================================================
# vLLM 최적화 모델 (권장) - 빠른 추론 속도
# ============================================================================

VLLM_MODELS = {
    "qwen-7b-vllm": {
        "name": "Qwen/Qwen2.5-Coder-7B-Instruct",
        "type": "vllm",
        "max_tokens": 4096,
        "context_length": 32768,
        "estimated_vram": "8GB",
        "tensor_parallel": 1,
        "gpu_memory_utilization": 0.9,
        "description": "vLLM 최적화 Qwen 7B - 가장 추천 (속도/성능 밸런스)",
        "inference_speed": "15-20x faster than HF",
        "best_for": "단일 GPU 환경"
    },
    "deepseek-7b-vllm": {
        "name": "deepseek-ai/deepseek-coder-7b-instruct-v1.5",
        "type": "vllm",
        "max_tokens": 4096,
        "context_length": 16384,
        "estimated_vram": "8GB",
        "tensor_parallel": 1,
        "gpu_memory_utilization": 0.9,
        "description": "vLLM 최적화 DeepSeek 7B - 코딩 특화",
        "inference_speed": "15-20x faster than HF",
        "best_for": "코드 생성 및 디버깅"
    },
    "codellama-13b-vllm": {
        "name": "codellama/CodeLlama-13b-Instruct-hf",
        "type": "vllm",
        "max_tokens": 4096,
        "context_length": 16384,
        "estimated_vram": "16GB",
        "tensor_parallel": 1,
        "gpu_memory_utilization": 0.9,
        "description": "vLLM 최적화 CodeLlama 13B - Meta 공식 모델",
        "inference_speed": "15-20x faster than HF",
        "best_for": "중형 GPU 환경 (RTX 3090/4090)"
    },
    "qwen-14b-vllm": {
        "name": "Qwen/Qwen2.5-Coder-14B-Instruct",
        "type": "vllm",
        "max_tokens": 4096,
        "context_length": 32768,
        "estimated_vram": "18GB",
        "tensor_parallel": 1,
        "gpu_memory_utilization": 0.9,
        "description": "vLLM 최적화 Qwen 14B - 고성능",
        "inference_speed": "15-20x faster than HF",
        "best_for": "고성능 단일 GPU"
    },
    "qwen-32b-vllm": {
        "name": "Qwen/Qwen2.5-Coder-32B-Instruct",
        "type": "vllm",
        "max_tokens": 8192,
        "context_length": 32768,
        "estimated_vram": "32GB+",
        "tensor_parallel": 2,
        "gpu_memory_utilization": 0.9,
        "description": "vLLM 최적화 Qwen 32B - 최고 성능",
        "inference_speed": "20-24x faster than HF",
        "best_for": "멀티 GPU 환경 (A100/H100)"
    },
    "deepseek-33b-vllm": {
        "name": "deepseek-ai/deepseek-coder-33b-instruct",
        "type": "vllm",
        "max_tokens": 8192,
        "context_length": 16384,
        "estimated_vram": "40GB+",
        "tensor_parallel": 2,
        "gpu_memory_utilization": 0.9,
        "description": "vLLM 최적화 DeepSeek 33B - 대형 코딩 모델",
        "inference_speed": "20-24x faster than HF",
        "best_for": "클라우드/멀티 GPU"
    }
}

# ============================================================================
# 로컬 환경용 가벼운 모델들 (CPU/낮은 GPU 사양) - HuggingFace Transformers
# ============================================================================

LOCAL_MODELS = {
    "qwen-1.5b": {
        "name": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
        "type": "huggingface",
        "max_tokens": 2048,
        "context_length": 8192,
        "estimated_ram": "3GB",
        "description": "가장 가벼운 코딩 특화 모델 (CPU 가능)",
        "use_case": "테스트 및 저사양 환경"
    },
    "deepseek-1.3b": {
        "name": "deepseek-ai/deepseek-coder-1.3b-instruct",
        "type": "huggingface",
        "max_tokens": 2048,
        "context_length": 16384,
        "estimated_ram": "3GB",
        "description": "DeepSeek 초경량 코딩 모델 (CPU 가능)",
        "use_case": "빠른 프로토타이핑"
    },
    "phi-2": {
        "name": "microsoft/phi-2",
        "type": "huggingface",
        "max_tokens": 2048,
        "context_length": 2048,
        "estimated_ram": "5GB",
        "description": "Microsoft의 소형 고성능 모델",
        "use_case": "일반 코딩 작업"
    },
    "tinyllama": {
        "name": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "type": "huggingface",
        "max_tokens": 2048,
        "context_length": 2048,
        "estimated_ram": "2GB",
        "description": "가장 가벼운 범용 모델 (CPU 가능)",
        "use_case": "최소 사양 환경"
    },
    "codellama-7b-q4": {
        "name": "TheBloke/CodeLlama-7B-Instruct-GGUF",
        "type": "ollama",
        "max_tokens": 4096,
        "context_length": 16384,
        "estimated_ram": "4GB (quantized)",
        "description": "4-bit 양자화된 CodeLlama 7B",
        "use_case": "Ollama 환경"
    }
}

# ============================================================================
# 클라우드/RunPod 환경용 큰 모델들 - vLLM 전용
# ============================================================================

RUNPOD_MODELS = {
    "qwen-32b": {
        "name": "Qwen/Qwen2.5-Coder-32B-Instruct",
        "type": "vllm",
        "max_tokens": 8192,
        "context_length": 32768,
        "estimated_vram": "32GB+",
        "tensor_parallel": 2,
        "gpu_memory_utilization": 0.95,
        "description": "고성능 코딩 모델 (2xA100 권장)",
        "deployment": "RunPod/AWS/GCP"
    },
    "codellama-34b": {
        "name": "codellama/CodeLlama-34b-Instruct-hf",
        "type": "vllm",
        "max_tokens": 16384,
        "context_length": 16384,
        "estimated_vram": "40GB+",
        "tensor_parallel": 2,
        "gpu_memory_utilization": 0.95,
        "description": "Meta CodeLlama 34B (2xA100 권장)",
        "deployment": "RunPod/AWS/GCP"
    },
    "deepseek-33b": {
        "name": "deepseek-ai/deepseek-coder-33b-instruct",
        "type": "vllm",
        "max_tokens": 8192,
        "context_length": 16384,
        "estimated_vram": "40GB+",
        "tensor_parallel": 2,
        "gpu_memory_utilization": 0.95,
        "description": "DeepSeek 대형 코딩 모델 (2xA100 권장)",
        "deployment": "RunPod/AWS/GCP"
    }
}

# ============================================================================
# 생성 파라미터 (Hint Generation Optimized)
# ============================================================================

# 창의적 힌트 생성 (기본값) - 동기 유발 및 상상력 자극
GENERATION_PARAMS = {
    "temperature": 0.7,  # 창의성과 일관성의 균형
    "top_p": 0.9,  # Nucleus sampling
    "top_k": 50,  # Top-K sampling
    "max_tokens": 512,  # 충분한 설명 길이
    "repetition_penalty": 1.1,  # 반복 억제
    "frequency_penalty": 0.3,  # vLLM용: 단어 반복 억제
    "presence_penalty": 0.1,  # vLLM용: 주제 다양성
    "do_sample": True
}

# 매우 창의적인 힌트 생성 (상상력 자극에 최적화)
CREATIVE_GENERATION_PARAMS = {
    "temperature": 0.85,  # 높은 창의성
    "top_p": 0.92,
    "top_k": 60,
    "max_tokens": 600,
    "repetition_penalty": 1.15,
    "frequency_penalty": 0.5,  # 더 다양한 표현
    "presence_penalty": 0.3,  # 더 다양한 주제
    "do_sample": True
}

# 소크라테스식 질문 생성 (질문 중심)
SOCRATIC_GENERATION_PARAMS = {
    "temperature": 0.75,
    "top_p": 0.9,
    "top_k": 50,
    "max_tokens": 400,
    "repetition_penalty": 1.2,  # 다양한 질문 유도
    "frequency_penalty": 0.4,
    "presence_penalty": 0.2,
    "do_sample": True
}

# 평가용 샘플링 파라미터 (더 결정론적, 일관성 중시)
EVAL_GENERATION_PARAMS = {
    "temperature": 0.3,  # 낮은 무작위성
    "top_p": 0.95,
    "max_tokens": 512,
    "repetition_penalty": 1.05,
    "frequency_penalty": 0.1,
    "presence_penalty": 0.0,
    "do_sample": True
}

# vLLM 서버 설정
VLLM_SERVER_CONFIG = {
    "default_url": "http://localhost:8000/v1",
    "timeout": 60,  # 요청 타임아웃 (초)
    "max_retries": 3,  # 최대 재시도 횟수
    "retry_delay": 1.0,  # 재시도 간격 (초)
}

# 모델 추천 (하드웨어별)
RECOMMENDED_MODELS = {
    "cpu_only": ["qwen-1.5b", "deepseek-1.3b", "tinyllama"],
    "gpu_4gb": ["qwen-1.5b", "deepseek-1.3b", "phi-2"],
    "gpu_8gb": ["qwen-7b-vllm", "deepseek-7b-vllm"],  # vLLM 권장
    "gpu_16gb": ["codellama-13b-vllm", "qwen-14b-vllm"],  # vLLM 권장
    "gpu_24gb": ["qwen-14b-vllm", "codellama-13b-vllm"],  # vLLM 권장
    "gpu_40gb+": ["qwen-32b-vllm", "deepseek-33b-vllm"],  # vLLM 멀티 GPU
}

# 사용 시나리오별 추천
USE_CASE_RECOMMENDATIONS = {
    "fastest_inference": "qwen-7b-vllm",  # 가장 빠른 추론
    "best_quality": "qwen-32b-vllm",  # 최고 품질
    "balanced": "deepseek-7b-vllm",  # 속도/품질 밸런스
    "low_resource": "qwen-1.5b",  # 저사양 환경
    "code_specialist": "deepseek-7b-vllm",  # 코드 특화
}
