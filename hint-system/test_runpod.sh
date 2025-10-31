#!/bin/bash
# RunPod 환경 완전성 테스트 스크립트

set -e

echo "============================================================"
echo "RunPod 환경 테스트"
echo "============================================================"

# 색상
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

passed=0
failed=0

check() {
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓${NC} $1"
        ((passed++))
    else
        echo -e "${RED}✗${NC} $1"
        ((failed++))
    fi
}

# 1. Python 버전
echo -e "\n${YELLOW}[1/10] Python 버전 확인${NC}"
python3 --version
check "Python 버전"

# 2. nvidia-smi (GPU)
echo -e "\n${YELLOW}[2/10] GPU 확인${NC}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
check "GPU 감지"

# 3. CUDA
echo -e "\n${YELLOW}[3/10] CUDA 버전 확인${NC}"
nvidia-smi | grep "CUDA Version"
check "CUDA 버전"

# 4. 가상환경 존재
echo -e "\n${YELLOW}[4/10] 가상환경 확인${NC}"
if [ -d "venv" ]; then
    echo "가상환경 존재"
    check "가상환경"
else
    echo "가상환경 없음"
    check "가상환경"
fi

# 5. vLLM 설치 (가상환경 활성화 후)
echo -e "\n${YELLOW}[5/10] vLLM 설치 확인${NC}"
if [ -d "venv" ]; then
    source venv/bin/activate
    python3 -c "import vllm; print(f'vLLM v{vllm.__version__}')" 2>/dev/null
    check "vLLM 설치"
else
    echo "가상환경 없음 - vLLM 확인 건너뜀"
    ((failed++))
fi

# 6. OpenAI 라이브러리
echo -e "\n${YELLOW}[6/10] OpenAI 라이브러리${NC}"
python3 -c "import openai; print(f'OpenAI v{openai.__version__}')" 2>/dev/null
check "OpenAI 라이브러리"

# 7. Gradio
echo -e "\n${YELLOW}[7/10] Gradio${NC}"
python3 -c "import gradio; print(f'Gradio v{gradio.__version__}')" 2>/dev/null
check "Gradio"

# 8. 프로젝트 파일
echo -e "\n${YELLOW}[8/10] 프로젝트 파일${NC}"
for file in app.py models/model_inference.py models/model_config.py start_vllm.sh run_app.sh; do
    if [ -f "$file" ]; then
        echo "  ✓ $file"
    else
        echo "  ✗ $file 없음"
        ((failed++))
    fi
done
((passed++))

# 9. 데이터 파일
echo -e "\n${YELLOW}[9/10] 데이터 파일${NC}"
if [ -f "data/problems_multi_solution.json" ]; then
    problem_count=$(python3 -c "import json; data=json.load(open('data/problems_multi_solution.json')); print(len(data))")
    echo "  문제 $problem_count개 로드"
    check "데이터 파일"
else
    echo "  data/problems_multi_solution.json 없음"
    check "데이터 파일"
fi

# 10. 실행 권한
echo -e "\n${YELLOW}[10/10] 실행 권한${NC}"
if [ -x "start_vllm.sh" ] && [ -x "run_app.sh" ]; then
    echo "  스크립트 실행 권한 있음"
    check "실행 권한"
else
    echo "  실행 권한 없음 - chmod +x 필요"
    check "실행 권한"
fi

# 결과
echo ""
echo "============================================================"
echo "테스트 결과"
echo "============================================================"
echo -e "통과: ${GREEN}$passed${NC}"
echo -e "실패: ${RED}$failed${NC}"

if [ $failed -eq 0 ]; then
    echo -e "\n${GREEN}✓ 모든 테스트 통과! RunPod에서 실행 가능합니다.${NC}"
    echo ""
    echo "다음 단계:"
    echo "1. vLLM 서버: ./start_vllm.sh"
    echo "2. Gradio 앱: ./run_app.sh"
    exit 0
else
    echo -e "\n${RED}✗ 일부 테스트 실패. 위의 오류를 해결하세요.${NC}"
    exit 1
fi
