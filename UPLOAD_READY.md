# 구글 드라이브 업로드 준비 완료

**준비 날짜:** 2025-01-30
**프로젝트:** 백준 힌트 생성 시스템
**총 용량:** ~2.05 MB

---

## ✅ 최종 체크리스트

### 1. 폴더 구조
```
5th-project_mvp/
├── README.md                    ✅ 프로젝트 메인 문서
├── SETUP_GUIDE.md               ✅ 환경 설정 가이드 (신규)
├── GOOGLE_DRIVE_GUIDE.md        ✅ 업로드 가이드
├── UPLOAD_READY.md              ✅ 이 파일
├── .env.example                 ✅ 환경 변수 예시 (신규)
├── config.py                    ✅ 환경 설정 자동화 (신규)
│
├── crawler/                     ✅ 크롤러
│   ├── README.md
│   └── crawlers/
│       ├── baekjoon_hybrid_crawler.py (config 연동)
│       ├── crawl_all_hybrid.py (config 연동)
│       ├── README_CRAWLING.md
│       └── requirements.txt (python-dotenv 추가)
│
├── hint-system/                 ✅ 힌트 시스템
│   ├── README.md
│   ├── app.py (config 연동)
│   ├── requirements.txt (python-dotenv 포함)
│   ├── models/
│   │   ├── __init__.py
│   │   ├── model_config.py
│   │   └── model_inference.py
│   ├── data/
│   │   └── problems_multi_solution.json (529개 문제)
│   └── evaluation/
│       └── results/             (빈 폴더)
│
├── docs/                        ✅ 문서
│   ├── PROMPT_FIX_V3.md
│   ├── MIGRATION_SUMMARY.md
│   └── PROMPT_IMPROVEMENT_LOG.md
│
└── sample-data/                 ✅ 샘플 데이터 (빈 폴더)
```

### 2. 캐시 정리
- ✅ `__pycache__` 폴더 삭제 완료
- ✅ `*.pyc` 파일 없음
- ✅ 불필요한 테스트 파일 제거

### 3. 문서 완성도
- ✅ 메인 README.md (프로젝트 개요, 빠른 시작)
- ✅ crawler/README.md (크롤러 사용법)
- ✅ hint-system/README.md (시스템 상세 가이드)
- ✅ docs/ (개발 문서 3개)

### 4. 핵심 파일
- ✅ app.py (Gradio UI)
- ✅ baekjoon_hybrid_crawler.py (크롤러)
- ✅ problems_multi_solution.json (529개 문제 데이터)
- ✅ requirements.txt (2개: 크롤러, 힌트시스템)

---

## 🚀 다음 단계

### 옵션 1: 폴더 직접 업로드 (권장)

1. **구글 드라이브 접속**
   - https://drive.google.com

2. **폴더 생성**
   - "새로 만들기" → "폴더"
   - 이름: `PlayData-Team5-BaekjoonHint`

3. **드래그 앤 드롭**
   - `baekjoon-hint-system` 폴더 전체를 구글 드라이브로 드래그
   - 용량: 2MB (업로드 시간 < 1분)

### 옵션 2: ZIP 압축 후 업로드

Windows PowerShell에서:
```powershell
cd "C:\Users\playdata2\Desktop\playdata\Workspace\팀프로젝트5"

# ZIP 생성
Compress-Archive -Path "baekjoon-hint-system" -DestinationPath "baekjoon-hint-system.zip"
```

그 다음 `baekjoon-hint-system.zip` 파일을 구글 드라이브에 업로드

---

## 📋 팀원 공유 설정

업로드 후:

1. **폴더 우클릭** → "공유"
2. **이메일 추가** (팀원 이메일)
3. **권한 설정:**
   - 편집자: 코드 수정 가능
   - 뷰어: 읽기만 가능

또는

1. **폴더 우클릭** → "링크 가져오기"
2. **액세스 권한:**
   - "링크가 있는 모든 사용자" (팀원에게 공유)

---

## 📦 포함된 내용

### 1. 크롤러
- 백준 문제 크롤링
- solved.ac API 통합
- 단계별 문제 수집

### 2. 힌트 시스템
- Gradio UI
- 소크라테스 학습법 기반 힌트 생성
- 다중 모델 지원 (Qwen, DeepSeek 등)
- Temperature 조절
- 코드 분석 (완료/누락 단계)
- 다중 풀이 자동 매칭

### 3. 데이터
- 529개 백준 문제
- 다중 풀이 지원
- 단계별 로직 분석

### 4. 문서
- 프롬프트 개선 로그 (V1~V4)
- 데이터 구조 마이그레이션 가이드
- 사용법 가이드

---

## 🎯 프로젝트 하이라이트

### V4 소크라테스 학습법
- ❌ 직접적 답 제공 금지
- ✅ 상황 제시 → 스스로 깨달음 유도
- ✅ 함수명/변수명 언급 자동 필터링

**예시:**
```
학생 코드: (반복 작업 수동으로 10번 작성)

❌ 나쁜 힌트: "count_repaint 함수를 만들어야 해"
✅ 좋은 힌트: "이 코드를 100번 써야 한다면 어떻게 할까?"
```

### 다중 풀이 지원
- 하나의 문제 → 여러 풀이 방법
- 사용자 코드와 자동 매칭
- 각 풀이에 맞는 힌트 제공

---

## 📞 문의

프로젝트 관련 문의: Team 5

**업로드 준비 완료일:** 2025-01-30
**버전:** 1.0.0
