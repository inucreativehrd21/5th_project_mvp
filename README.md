# 백준 문제 힌트 생성 시스템

소크라테스 학습법 기반 코딩 힌트 제공 시스템

## 📁 프로젝트 구조

```
5th-project_mvp/
├── .env.example         # 환경 설정 예시 파일 ⭐ 중요!
├── config.py            # 환경 설정 자동화 ⭐ 신규!
├── crawler/             # 백준 문제 크롤러
│   └── crawlers/
│       ├── baekjoon_hybrid_crawler.py
│       ├── crawl_all_hybrid.py
│       └── requirements.txt
├── hint-system/         # 힌트 생성 시스템 (메인)
│   ├── app.py           # Gradio UI 메인 앱
│   ├── models/          # 모델 추론 로직
│   ├── data/            # 문제 데이터 (529개 고유 문제)
│   │   └── problems_multi_solution.json
│   ├── evaluation/      # 평가 결과 저장
│   └── requirements.txt
├── docs/                # 프로젝트 문서
│   ├── PROMPT_FIX_V3.md
│   ├── MIGRATION_SUMMARY.md
│   └── PROMPT_IMPROVEMENT_LOG.md
├── sample-data/         # 크롤링 샘플 데이터
└── README.md            # 이 파일
```

## 🚀 빠른 시작

### ⚠️ 필수: 환경 설정 (.env 파일 생성)

**처음 다운로드한 경우 반드시 아래 단계를 먼저 수행하세요:**

```bash
# 1. .env.example 파일을 .env로 복사
cp .env.example .env

# 2. .env 파일을 열어서 본인 환경에 맞게 경로 수정
# Windows 예시:
PROJECT_ROOT=C:\Users\YourName\Desktop\5th-project_mvp

# Mac/Linux 예시:
PROJECT_ROOT=/Users/YourName/Desktop/5th-project_mvp
```

**왜 필요한가요?**
- 각자의 컴퓨터 환경이 다르기 때문에 **경로를 자동으로 설정**하기 위함
- `.env` 파일에 한 번만 경로를 입력하면, 모든 파일에서 자동으로 적용됨
- 팀원끼리 코드를 공유할 때 **경로 충돌 문제 해결**

### 1단계: 크롤러 실행 (선택사항)

백준에서 문제 데이터 크롤링 (이미 준비된 데이터가 있으면 건너뛰기):

```bash
# 가상환경 설정
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac

# 크롤러 의존성 설치
cd crawler/crawlers
pip install -r requirements.txt

# 크롤링 실행 (자동으로 config.py에서 경로 읽음)
python crawl_all_hybrid.py
```

**크롤러 기능:**
- 백준 단계별 문제 목록 수집
- 문제 설명, 입출력 예제 크롤링
- solved.ac API로 난이도/태그 추가
- JSON 형식으로 저장 (경로: `crawler/data/raw/`)

### 2단계: 힌트 시스템 실행

```bash
cd hint-system

# 가상환경이 없으면 생성 (위에서 만들었으면 건너뛰기)
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac

# 패키지 설치
pip install -r requirements.txt

# 앱 실행 (자동으로 config.py에서 경로 읽음)
python app.py
```

브라우저에서 `http://localhost:7860` 접속

**실행 시 자동으로 표시되는 정보:**
```
프로젝트 환경 설정
============================================================
프로젝트 루트: C:\Users\...\5th-project_mvp
데이터 파일: hint-system\data\problems_multi_solution.json
크롤러 출력: crawler\data\raw
평가 결과: hint-system\evaluation\results
로그 파일: logs\app.log
============================================================
```

## 📚 주요 기능

### 1. 백준 문제 크롤링
- **단계별 문제 목록** 수집 (1~68단계)
- **문제 상세 정보**: 제목, 설명, 입출력 예제
- **solved.ac API 통합**: 난이도(Level), 태그(분류)
- **JSON 저장**: `problems_hybrid_step_X_to_Y.json` 형식

### 2. 힌트 생성 시스템

#### 소크라테스 학습법
- **직접 답을 주지 않고** 학생이 스스로 생각하게 유도
- 문제 상황 제시 → 학생이 해결책 도출

**예시:**
```
❌ 나쁜 힌트: "함수를 정의하려면 def 키워드가 필요해"
✅ 좋은 힌트: "이 계산을 100번 써야 한다면 코드를 100번 복사할 건가?"
```

#### 코드 분석
- **자동 분석**: 사용자가 완료한 부분 vs 누락된 부분
- **구조 파악**: 함수, 반복문, 조건문, 출력 여부 체크
- **다음 단계 제시**: N번째 Logic 완료 → N+1번째 힌트

#### 다중 모델 비교
- **Qwen2.5-Coder-1.5B**: 빠르고 가벼움
- **DeepSeek-Coder-1.3B**: 코드 이해 특화
- **TinyLlama-1.1B**: 경량 모델
- 동시에 여러 모델 힌트 비교 가능

#### Temperature 조절
- **0.1**: 일관된 힌트 (거의 같은 결과)
- **0.5**: 적당한 다양성 (권장)
- **1.0**: 창의적이지만 불안정

### 3. 다중 풀이 지원

하나의 문제에 여러 풀이 방법 제공:

**예시: A+B 문제**
- 풀이 1: 각각 입력 (`A = int(input())`)
- 풀이 2: split 활용 (`A, B = map(int, input().split())`)

**자동 매칭:**
사용자 코드와 가장 유사한 풀이 자동 선택

## 📊 데이터 구조

### problems_multi_solution.json

```json
{
  "problem_id": "1000",
  "title": "A+B",
  "level": 1,
  "tags": ["구현", "사칙연산", "수학"],
  "description": "두 정수 A와 B를 입력받은 다음, A+B를 출력하는 프로그램을 작성하시오.",
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

**통계:**
- 총 529개 고유 문제
- 475개 단일 풀이
- 54개 다중 풀이 (평균 4.2개 풀이)

## 🛠️ 기술 스택

### 크롤러
- **BeautifulSoup4**: HTML 파싱
- **requests**: HTTP 통신
- **solved.ac API**: 문제 메타데이터

### 힌트 시스템
- **Gradio**: 웹 UI 프레임워크
- **Transformers**: 모델 로딩 (HuggingFace)
- **PyTorch**: 딥러닝 프레임워크
- **BitsAndBytes**: 4-bit 양자화 (메모리 절약)

### 지원 모델
| 모델 | 파라미터 | 양자화 | 메모리 |
|------|---------|--------|--------|
| Qwen2.5-Coder-1.5B | 1.5B | FP16 | ~3GB |
| DeepSeek-Coder-1.3B | 1.3B | FP16 | ~2.6GB |
| TinyLlama-1.1B | 1.1B | FP16 | ~2.2GB |
| Phi-2 | 2.7B | 4-bit | ~1.7GB |
| CodeLlama-7B | 7B | 4-bit | ~4GB |

## 📖 상세 문서

### 시스템 문서
- [힌트 시스템 사용법](hint-system/README.md)
- [크롤러 사용법](crawler/README.md)

### 개발 문서
- [프롬프트 개선 로그 V3](docs/PROMPT_FIX_V3.md) - 최신
- [데이터 구조 변경](docs/MIGRATION_SUMMARY.md)
- [프롬프트 개선 전체 로그](docs/PROMPT_IMPROVEMENT_LOG.md)

## 🎯 프롬프트 엔지니어링

### V1: 기본 컨텍스트
- 코드 구조 분석 추가
- 완료/누락 단계 파악

### V2: 소크라테스 강화
- 정답 코드 언급 금지
- 직접 답 요구 차단

### V3: 함수명 차단 (최신)
- snake_case 자동 필터링
- 예시 제거 (모델이 따라하지 않도록)
- 진짜 소크라테스식 질문

**상세:** [PROMPT_FIX_V3.md](docs/PROMPT_FIX_V3.md)

## 🔧 시스템 요구사항

### 최소 사양
- **RAM**: 8GB 이상
- **GPU**: 선택사항 (CPU로도 작동, 느림)
- **저장공간**: 10GB 이상

### 권장 사양
- **RAM**: 12GB 이상
- **GPU**: NVIDIA 6GB+ VRAM (RTX 3060 이상)
- **저장공간**: 20GB 이상

### AWS/RunPod 배포 시
- **vLLM 사용 가능** (20-60x 속도 향상)
- **추천 인스턴스**: g4dn.xlarge 이상

## 📝 사용 예시

### 1. 문제 선택
![문제 선택](docs/images/step1.png)

### 2. 코드 작성
```python
# 학생이 입력만 작성
N, M = map(int, input().split())
board = []
for _ in range(N):
    board.append(input())
```

### 3. 힌트 받기
**Temperature 0.3, Qwen2.5-Coder 모델:**
```
"이 계산을 64번 반복해야 한다면 코드를 64번 복사할 건가?"
```

### 4. 평가
- 1~5점 평가
- 코멘트 작성
- 결과 저장 (`evaluation/results/`)

## 🚨 주의사항

1. **함수명/변수명 언급 방지**
   - snake_case 패턴 자동 필터링
   - 일반 함수(input, print)는 허용

2. **모델 크기**
   - 소형 모델(1.5B)은 불안정할 수 있음
   - Temperature 낮춰서 사용 권장

3. **메모리 관리**
   - 여러 모델 동시 로드 시 메모리 부족 가능
   - 필요한 모델만 선택

## 📄 라이선스

Educational use only

## 👥 기여자

Team 5 - PlayData AI 부트캠프

## 📧 문의

- GitHub Issues
- Email: [your-email]

---

**마지막 업데이트:** 2025-01-30
**버전:** 1.0.0
