# 힌트 생성 시스템

소크라테스 학습법 기반 코딩 힌트 제공 시스템

## 🚀 빠른 시작

```bash
# 1. 패키지 설치
pip install -r requirements.txt

# 2. 앱 실행
python app.py

# 3. 브라우저 접속
http://localhost:7860
```

## 📁 구조

```
hint-system/
├── app.py                    # Gradio UI 메인 앱
├── models/
│   ├── model_inference.py   # 모델 추론 로직
│   └── model_config.py      # 모델 설정
├── data/
│   └── problems_multi_solution.json  # 529개 문제
├── evaluation/
│   └── results/             # 평가 결과 저장
└── requirements.txt         # 패키지 목록
```

## 🎯 사용 흐름

### 1. 문제 선택
- 529개 백준 문제 중 선택
- Level, 태그로 필터링 가능

### 2. 코드 작성
```python
# 예시: A+B 문제 - 입력만 작성
N, M = map(int, input().split())
board = []
for _ in range(N):
    board.append(input())
```

### 3. 모델 선택 & Temperature 조절
- **모델 선택**: 1~5개 동시 비교 가능
- **Temperature**: 0.1 (일관) ~ 1.0 (창의)

### 4. 힌트 받기
**자동 분석:**
```
✅ 완료: 입력 받기, 반복문
❌ 누락: 함수 정의, 조건문, 출력
🎯 다음: 함수 정의
```

**힌트 (소크라테스식):**
```
"이 계산을 100번 반복해야 한다면 코드를 100번 복사할 건가?"
```

### 5. 평가
- 1~5점 평가
- 코멘트 작성
- 자동 저장: `evaluation/results/`

## 🧠 소크라테스 학습법

### 기본 원칙
> 답을 직접 주지 말고, 학생이 스스로 생각하게 유도

### 질문 방식

**❌ 나쁜 힌트 (직접 답 유도):**
```
"함수를 정의하려면 def 키워드가 필요해"
→ 답: def (너무 직접적)
```

**✅ 좋은 힌트 (소크라테스식):**
```
"이 작업을 100번 써야 한다면 코드를 100번 복사할 건가?"
→ 학생: "아니, 재사용하려면... 함수!" (스스로 깨달음)
```

### 구현 방식

1. **코드 분석**: 완료/누락 자동 파악
2. **문제 상황 제시**: "100번 복사?"
3. **학생 사고 유도**: 스스로 해결책 생각
4. **함수명 언급 금지**: snake_case 자동 필터링

## 🤖 모델 정보

### 지원 모델

| 모델 | 파라미터 | 양자화 | 메모리 | 특징 |
|------|---------|--------|--------|------|
| Qwen2.5-Coder-1.5B | 1.5B | FP16 | ~3GB | 가장 빠름 |
| DeepSeek-Coder-1.3B | 1.3B | FP16 | ~2.6GB | 코드 이해 특화 |
| TinyLlama-1.1B | 1.1B | FP16 | ~2.2GB | 가장 가벼움 |
| Phi-2 | 2.7B | 4-bit | ~1.7GB | 품질 좋음 |
| CodeLlama-7B | 7B | 4-bit | ~4GB | 고품질 (느림) |

### 모델 선택 가이드

**빠른 테스트:**
- Qwen2.5-Coder-1.5B (권장)
- Temperature 0.3

**품질 우선:**
- Phi-2 또는 CodeLlama-7B
- Temperature 0.5

**여러 관점 비교:**
- 3~5개 모델 동시 실행
- 다양한 힌트 비교

## ⚙️ Temperature 가이드

| 값 | 특징 | 추천 상황 |
|----|------|-----------|
| 0.1 | 거의 같은 힌트 | 일관성 필요 |
| 0.3 | 약간의 다양성 | **기본 권장** |
| 0.5 | 적당한 창의성 | 다양한 힌트 |
| 0.8 | 매우 창의적 | 실험적 |
| 1.0 | 예측 불가 | 권장 안 함 |

## 📊 데이터 구조

### problems_multi_solution.json

```json
{
  "problem_id": "1000",
  "title": "A+B",
  "level": 1,
  "tags": ["구현", "사칙연산"],
  "solutions": [
    {
      "solution_id": 1,
      "solution_name": "풀이 1: 각각 입력",
      "solution_code": "A = int(input())\nB = int(input())\nprint(A + B)",
      "logic_steps": [
        {
          "step_id": 1,
          "goal": "첫 번째 정수 A 입력받기",
          "code_pattern": "A = int(input())",
          "socratic_hint": "첫 번째 숫자를 정수로 입력받아 변수에 저장하려면?"
        }
      ]
    },
    {
      "solution_id": 2,
      "solution_name": "풀이 2: split 활용",
      "solution_code": "A, B = map(int, input().split())\nprint(A + B)",
      "logic_steps": [...]
    }
  ]
}
```

### 다중 풀이 자동 매칭

사용자 코드 패턴을 분석하여 가장 유사한 풀이 자동 선택:

```python
# 사용자가 split() 사용
A, B = map(int, input().split())
→ "풀이 2: split 활용" 매칭

# 사용자가 각각 입력
A = int(input())
B = int(input())
→ "풀이 1: 각각 입력" 매칭
```

**매칭 기준:**
- split 사용 여부
- input() 개수
- map 사용 여부
- 반복문/조건문 사용

## 🔧 고급 설정

### 모델 추가

```python
# app.py 수정
default_models = [
    {
        "name": "새 모델",
        "path": "org/model-name",
        "quantize": False,
        "size": "1.5B"
    }
]
```

### 프롬프트 수정

```python
# app.py: _create_analysis_prompt() 메서드
prompt = f"""학생 코드:
{user_code}

다음 단계: {next_step_goal}

질문:"""
```

### 필터 조정

```python
# models/model_inference.py
# 최소 힌트 길이
if len(sentence) < 30:  # 30자 → 원하는 길이로 변경
    continue
```

## 🚨 문제 해결

### 1. 모델 로딩 실패
```
[ERROR] 모델 로드 실패
```

**해결:**
- RAM 부족: 더 작은 모델 사용
- 네트워크: HuggingFace 연결 확인
- 캐시 삭제: `~/.cache/huggingface` 삭제

### 2. 힌트가 이상함
```
"count_repaint 함수를 만들어야 할까?"
```

**원인:** 함수명 언급 (필터 실패)

**해결:**
- Temperature 낮추기 (0.2~0.3)
- 다른 모델 시도
- 프롬프트 확인

### 3. 메모리 부족
```
CUDA out of memory
```

**해결:**
- 4-bit 양자화 모델 사용 (Phi-2, CodeLlama-7B)
- 모델 개수 줄이기 (1~2개만)
- CPU 모드 사용

### 4. Gradio 자동 실행 안 됨

**확인:**
```python
# app.py 마지막 줄
demo.launch(
    server_port=7860,
    inbrowser=True  # 브라우저 자동 열기
)
```

**수동 접속:**
```
http://localhost:7860
```

## 📈 성능 최적화

### CPU만 있는 경우
- **모델**: Qwen2.5-Coder-1.5B, TinyLlama
- **개수**: 1~2개만
- **예상 시간**: 힌트 1개당 10~30초

### GPU 있는 경우 (6GB+)
- **모델**: 모든 모델 가능
- **개수**: 3~5개 동시
- **예상 시간**: 힌트 1개당 2~5초

### 클라우드 (vLLM)
- **속도**: 20~60x 빠름
- **추천**: AWS g4dn.xlarge, RunPod
- **비용**: 시간당 $0.5~1.0

## 📝 평가 결과

### 저장 위치
```
evaluation/results/
└── evaluation_YYYYMMDD_HHMMSS.json
```

### 형식
```json
{
  "problem_id": "1000",
  "problem_title": "A+B",
  "timestamp": "2025-01-30T16:30:00",
  "ratings": {
    "Qwen2.5-Coder-1.5B": {
      "hint": "이 계산을 100번 써야 한다면?",
      "rating": 5,
      "generation_time": 2.3
    }
  },
  "comments": "매우 좋은 소크라테스식 힌트"
}
```

## 🎓 교육 활용

### 1. 개인 학습
- 문제 풀다가 막히면 힌트 요청
- 여러 모델 비교하며 학습

### 2. 수업 활용
- 학생 코드 분석
- 적절한 힌트 제공
- 학생 스스로 해결 유도

### 3. 평가 시스템
- 모델 힌트 품질 평가
- 데이터 축적
- 프롬프트 개선

---

**마지막 업데이트:** 2025-01-30
**버전:** 1.0.0
