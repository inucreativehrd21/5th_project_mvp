#!/bin/bash
# RunPod 올인원 배포 스크립트
# 설치 → vLLM 서버 → Gradio 앱을 한 번에 실행
#
# 사용법:
#   bash deploy_runpod.sh          # 전체 설치 및 실행
#   bash deploy_runpod.sh --skip-install  # 설치 건너뛰기

set -e

# 색상
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}"
echo "============================================================"
echo "   vLLM 힌트 생성 시스템 - RunPod 자동 배포"
echo "============================================================"
echo -e "${NC}"

SKIP_INSTALL=false
if [ "$1" == "--skip-install" ]; then
    SKIP_INSTALL=true
fi

# ============================================================================
# 1. 환경 확인
# ============================================================================
echo -e "${YELLOW}[1/5] 환경 확인 중...${NC}"

# GPU 확인
if ! command -v nvidia-smi &> /dev/null; then
    echo -e "${RED}✗ NVIDIA GPU를 찾을 수 없습니다.${NC}"
    exit 1
fi

GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
GPU_MEMORY=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)

echo -e "${GREEN}✓ GPU: $GPU_NAME${NC}"
echo -e "${GREEN}✓ VRAM: ${GPU_MEMORY}MB${NC}"
echo -e "${GREEN}✓ GPU 개수: ${GPU_COUNT}${NC}"

# Python 확인
python3 --version || { echo -e "${RED}✗ Python3 필요${NC}"; exit 1; }

# ============================================================================
# 2. 자동 설치 (선택적)
# ============================================================================
if [ "$SKIP_INSTALL" = false ]; then
    echo -e "\n${YELLOW}[2/5] 의존성 설치 중... (5-10분 소요)${NC}"

    # 가상환경
    if [ ! -d "venv" ]; then
        python3 -m venv venv
        echo -e "${GREEN}✓ 가상환경 생성${NC}"
    fi

    source venv/bin/activate

    # pip 업그레이드
    pip install --upgrade pip setuptools wheel -q

    # CUDA 버전 감지
    cuda_version=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}' | cut -d. -f1,2)
    echo "CUDA: $cuda_version"

    # PyTorch
    if [[ "$cuda_version" == "12.1" ]] || [[ "$cuda_version" > "12.1" ]]; then
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 -q
    else
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 -q
    fi

    # vLLM
    if [[ "$cuda_version" == "12.1" ]] || [[ "$cuda_version" > "12.1" ]]; then
        pip install vllm --extra-index-url https://download.pytorch.org/whl/cu121 -q
    else
        pip install vllm --extra-index-url https://download.pytorch.org/whl/cu118 -q
    fi

    # 기타
    pip install openai python-dotenv requests gradio transformers accelerate sentencepiece protobuf -q

    echo -e "${GREEN}✓ 설치 완료${NC}"

    # .env 파일 생성
    if [ ! -f ".env" ]; then
        cat > .env <<EOF
PROJECT_ROOT=/workspace/hint-system
DATA_FILE_PATH=hint-system/data/problems_multi_solution.json
EVALUATION_RESULTS_DIR=hint-system/evaluation/results
VLLM_SERVER_URL=http://localhost:8000/v1
VLLM_PORT=8000
VLLM_GPU_MEMORY_UTILIZATION=0.9
VLLM_MAX_MODEL_LEN=4096
DEFAULT_TEMPERATURE=0.7
EOF
        echo -e "${GREEN}✓ .env 파일 생성${NC}"
    fi
else
    echo -e "\n${YELLOW}[2/5] 설치 건너뛰기${NC}"
    source venv/bin/activate
fi

# ============================================================================
# 3. GPU 기반 모델 자동 선택
# ============================================================================
echo -e "\n${YELLOW}[3/5] 최적 모델 선택 중...${NC}"

if [ "$GPU_MEMORY" -ge 40000 ]; then
    MODEL="Qwen/Qwen2.5-Coder-32B-Instruct"
    TENSOR_PARALLEL=2
    echo -e "${GREEN}✓ 모델: Qwen 32B (대용량 GPU)${NC}"
elif [ "$GPU_MEMORY" -ge 24000 ]; then
    MODEL="Qwen/Qwen2.5-Coder-14B-Instruct"
    TENSOR_PARALLEL=1
    echo -e "${GREEN}✓ 모델: Qwen 14B (고성능)${NC}"
elif [ "$GPU_MEMORY" -ge 16000 ]; then
    MODEL="codellama/CodeLlama-13b-Instruct-hf"
    TENSOR_PARALLEL=1
    echo -e "${GREEN}✓ 모델: CodeLlama 13B${NC}"
else
    MODEL="Qwen/Qwen2.5-Coder-7B-Instruct"
    TENSOR_PARALLEL=1
    echo -e "${GREEN}✓ 모델: Qwen 7B${NC}"
fi

# 멀티 GPU
if [ "$GPU_COUNT" -gt 1 ] && [[ "$MODEL" == *"32B"* || "$MODEL" == *"33B"* ]]; then
    TENSOR_PARALLEL=$GPU_COUNT
    echo -e "${GREEN}✓ 멀티 GPU: Tensor Parallel = $TENSOR_PARALLEL${NC}"
fi

# ============================================================================
# 4. vLLM 서버 시작 (백그라운드)
# ============================================================================
echo -e "\n${YELLOW}[4/5] vLLM 서버 시작 중...${NC}"

mkdir -p logs

# tmux 세션 확인 및 생성
if command -v tmux &> /dev/null; then
    # 기존 세션 종료
    tmux kill-session -t vllm 2>/dev/null || true

    # vLLM 서버 시작 (tmux)
    tmux new-session -d -s vllm "python -m vllm.entrypoints.openai.api_server \
        --model $MODEL \
        --host 0.0.0.0 \
        --port 8000 \
        --gpu-memory-utilization 0.9 \
        --max-model-len 4096 \
        --tensor-parallel-size $TENSOR_PARALLEL \
        --trust-remote-code \
        --dtype auto \
        2>&1 | tee logs/vllm_server_\$(date +%Y%m%d_%H%M%S).log"

    echo -e "${GREEN}✓ vLLM 서버 시작됨 (tmux 세션: vllm)${NC}"
    echo -e "${BLUE}  확인: tmux attach -t vllm (Ctrl+B, D로 나가기)${NC}"
else
    # tmux 없으면 nohup 사용
    nohup python -m vllm.entrypoints.openai.api_server \
        --model $MODEL \
        --host 0.0.0.0 \
        --port 8000 \
        --gpu-memory-utilization 0.9 \
        --max-model-len 4096 \
        --tensor-parallel-size $TENSOR_PARALLEL \
        --trust-remote-code \
        --dtype auto \
        > logs/vllm_server_$(date +%Y%m%d_%H%M%S).log 2>&1 &

    echo -e "${GREEN}✓ vLLM 서버 시작됨 (백그라운드)${NC}"
    echo -e "${BLUE}  로그: tail -f logs/vllm_server_*.log${NC}"
fi

# 서버 대기 (최대 5분)
echo -e "\n${YELLOW}모델 다운로드 및 로딩 중... (5-15분, 첫 실행만)${NC}"

for i in {1..300}; do
    if curl -s http://localhost:8000/v1/models > /dev/null 2>&1; then
        echo -e "${GREEN}✓ vLLM 서버 준비 완료!${NC}"
        break
    fi

    if [ $((i % 10)) -eq 0 ]; then
        echo "  대기 중... ($i초 경과)"
    fi
    sleep 1
done

# 서버 확인
if ! curl -s http://localhost:8000/v1/models > /dev/null 2>&1; then
    echo -e "${RED}✗ vLLM 서버 시작 실패${NC}"
    echo -e "${YELLOW}로그 확인: tail -f logs/vllm_server_*.log${NC}"
    exit 1
fi

# ============================================================================
# 5. Gradio 앱 시작
# ============================================================================
echo -e "\n${YELLOW}[5/5] Gradio 앱 시작 중...${NC}"

# RunPod 환경 감지
if [ -n "$RUNPOD_POD_ID" ] || [ -n "$PUBLIC_URL" ]; then
    echo -e "${GREEN}✓ RunPod 환경 감지${NC}"
    APP_ARGS="--server-name 0.0.0.0 --server-port 7860 --share"
else
    APP_ARGS="--server-port 7860"
fi

echo -e "${GREEN}✓ Gradio 앱 시작${NC}"
echo -e "${BLUE}"
echo "============================================================"
echo "   배포 완료!"
echo "============================================================"
echo -e "${NC}"
echo ""
echo -e "${GREEN}vLLM 서버:${NC} http://localhost:8000"
echo -e "${GREEN}Gradio 앱:${NC} http://localhost:7860"
echo ""
echo -e "${YELLOW}유용한 명령어:${NC}"
echo "  - tmux attach -t vllm     # vLLM 서버 로그 보기"
echo "  - tmux kill-session -t vllm  # vLLM 서버 종료"
echo "  - tail -f logs/vllm_server_*.log  # vLLM 로그"
echo "  - nvidia-smi     # GPU 모니터링"
echo ""

# Gradio 실행
python app.py $APP_ARGS
