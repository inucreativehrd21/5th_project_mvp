#!/bin/bash
# RunPod 배포 전 최종 검수 스크립트
# Linux 환경 전용

set -e

# 색상
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}============================================================"
echo "RunPod 배포 전 최종 검수"
echo "============================================================${NC}"

passed=0
failed=0
warnings=0

check() {
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓${NC} $1"
        ((passed++))
        return 0
    else
        echo -e "${RED}✗${NC} $1"
        ((failed++))
        return 1
    fi
}

warn() {
    echo -e "${YELLOW}⚠${NC} $1"
    ((warnings++))
}

# ============================================================================
# 1. 파일 존재 확인
# ============================================================================
echo -e "\n${BLUE}[1/7] 필수 파일 확인${NC}"

required_files=(
    "app.py"
    "models/model_inference.py"
    "models/model_config.py"
    "models/__init__.py"
    "setup_runpod.sh"
    "start_vllm.sh"
    "run_app.sh"
    "test_imports.py"
    "test_vllm_integration.py"
    "test_runpod.sh"
    ".env.example"
    "data/problems_multi_solution.json"
    "requirements.txt"
)

for file in "${required_files[@]}"; do
    if [ -f "$file" ]; then
        echo -e "  ${GREEN}✓${NC} $file"
    else
        echo -e "  ${RED}✗${NC} $file 없음"
        ((failed++))
    fi
done

# ============================================================================
# 2. 실행 권한 확인
# ============================================================================
echo -e "\n${BLUE}[2/7] 실행 권한 확인${NC}"

executable_files=(
    "setup_runpod.sh"
    "start_vllm.sh"
    "run_app.sh"
    "test_runpod.sh"
)

for file in "${executable_files[@]}"; do
    if [ -x "$file" ]; then
        echo -e "  ${GREEN}✓${NC} $file"
    else
        echo -e "  ${YELLOW}⚠${NC} $file (chmod +x 필요)"
        chmod +x "$file" 2>/dev/null || warn "권한 설정 실패"
    fi
done

# ============================================================================
# 3. Python 코드 문법 검사
# ============================================================================
echo -e "\n${BLUE}[3/7] Python 코드 문법 검사${NC}"

python_files=(
    "app.py"
    "models/model_inference.py"
    "models/model_config.py"
    "test_imports.py"
    "test_vllm_integration.py"
    "vllm_server.py"
)

for file in "${python_files[@]}"; do
    if [ -f "$file" ]; then
        python3 -m py_compile "$file" 2>/dev/null
        if [ $? -eq 0 ]; then
            echo -e "  ${GREEN}✓${NC} $file"
        else
            echo -e "  ${RED}✗${NC} $file (문법 오류)"
            ((failed++))
        fi
    fi
done

# ============================================================================
# 4. Import 테스트
# ============================================================================
echo -e "\n${BLUE}[4/7] Import 테스트${NC}"

python3 -c "
import sys
sys.path.insert(0, '..')

try:
    from config import Config
    print('  ✓ config.py')
except Exception as e:
    print(f'  ✗ config.py: {e}')
    sys.exit(1)

try:
    from models.model_config import VLLM_MODELS, LOCAL_MODELS
    print('  ✓ model_config.py')
except Exception as e:
    print(f'  ✗ model_config.py: {e}')
    sys.exit(1)

try:
    from models.model_inference import ModelManager, VLLMInference
    print('  ✓ model_inference.py')
except Exception as e:
    print(f'  ✗ model_inference.py: {e}')
    sys.exit(1)
" 2>/dev/null
check "Import 구조"

# ============================================================================
# 5. 데이터 파일 검증
# ============================================================================
echo -e "\n${BLUE}[5/7] 데이터 파일 검증${NC}"

if [ -f "data/problems_multi_solution.json" ]; then
    # JSON 유효성 검사
    python3 -c "
import json
with open('data/problems_multi_solution.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
    print(f'  ✓ JSON 유효 ({len(data)}개 문제)')
" 2>/dev/null
    check "데이터 파일"
else
    echo -e "  ${RED}✗${NC} data/problems_multi_solution.json 없음"
    ((failed++))
fi

# ============================================================================
# 6. 환경 변수 템플릿 검증
# ============================================================================
echo -e "\n${BLUE}[6/7] 환경 변수 템플릿 검증${NC}"

if [ -f ".env.example" ]; then
    # Linux 경로 확인
    if grep -q "/workspace/hint-system" ".env.example"; then
        echo -e "  ${GREEN}✓${NC} Linux 경로 설정됨"
    else
        echo -e "  ${YELLOW}⚠${NC} Windows 경로 포함 - Linux 경로로 변경 필요"
        ((warnings++))
    fi

    # 필수 변수 확인
    required_vars=(
        "PROJECT_ROOT"
        "DATA_FILE_PATH"
        "VLLM_SERVER_URL"
        "VLLM_MODEL"
        "DEFAULT_TEMPERATURE"
    )

    for var in "${required_vars[@]}"; do
        if grep -q "^$var=" ".env.example"; then
            echo -e "  ${GREEN}✓${NC} $var"
        else
            echo -e "  ${RED}✗${NC} $var 없음"
            ((failed++))
        fi
    done
else
    echo -e "  ${RED}✗${NC} .env.example 없음"
    ((failed++))
fi

# ============================================================================
# 7. 스크립트 내용 검증
# ============================================================================
echo -e "\n${BLUE}[7/7] 스크립트 내용 검증${NC}"

# setup_runpod.sh 검증
if grep -q "vllm" "setup_runpod.sh"; then
    echo -e "  ${GREEN}✓${NC} setup_runpod.sh: vLLM 설치 포함"
else
    echo -e "  ${RED}✗${NC} setup_runpod.sh: vLLM 설치 누락"
    ((failed++))
fi

# start_vllm.sh 검증
if grep -q "nvidia-smi" "start_vllm.sh"; then
    echo -e "  ${GREEN}✓${NC} start_vllm.sh: GPU 감지 포함"
else
    echo -e "  ${RED}✗${NC} start_vllm.sh: GPU 감지 누락"
    ((failed++))
fi

if grep -q "0.0.0.0" "start_vllm.sh"; then
    echo -e "  ${GREEN}✓${NC} start_vllm.sh: 0.0.0.0 바인딩"
else
    echo -e "  ${YELLOW}⚠${NC} start_vllm.sh: localhost만 바인딩 (외부 접근 불가)"
    ((warnings++))
fi

# run_app.sh 검증
if grep -q "RUNPOD" "run_app.sh" || grep -q "PUBLIC_URL" "run_app.sh"; then
    echo -e "  ${GREEN}✓${NC} run_app.sh: RunPod 환경 감지"
else
    echo -e "  ${YELLOW}⚠${NC} run_app.sh: RunPod 환경 감지 없음"
    ((warnings++))
fi

# ============================================================================
# 결과 요약
# ============================================================================
echo -e "\n${BLUE}============================================================"
echo "검수 결과 요약"
echo "============================================================${NC}"

total=$((passed + failed))
echo -e "${GREEN}통과:${NC} $passed"
echo -e "${RED}실패:${NC} $failed"
echo -e "${YELLOW}경고:${NC} $warnings"

if [ $failed -eq 0 ]; then
    echo -e "\n${GREEN}✅ 모든 검수 통과! RunPod 배포 준비 완료.${NC}"
    echo ""
    echo "다음 단계:"
    echo "1. RunPod Pod 생성 (GPU: RTX 4090 권장)"
    echo "2. git clone <repository>"
    echo "3. cd hint-system/hint-system"
    echo "4. bash setup_runpod.sh"
    echo "5. ./start_vllm.sh (터미널 1)"
    echo "6. ./run_app.sh (터미널 2)"
    echo ""
    echo "예상 비용: $0.39/시간 (RTX 4090)"
    echo "예상 성능: 0.5초/힌트 (17배 빠름)"
    exit 0
else
    echo -e "\n${RED}✗ 검수 실패 ($failed개 오류)${NC}"
    echo "위의 오류를 수정한 후 다시 실행하세요:"
    echo "  bash pre_deployment_check.sh"
    exit 1
fi
