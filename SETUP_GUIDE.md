# 🔧 환경 설정 가이드

프로젝트를 처음 다운로드한 경우 **반드시** 이 가이드를 따라 설정하세요.

---

## ⚙️ 1단계: .env 파일 생성

### Windows

```powershell
# 프로젝트 루트 폴더로 이동
cd C:\Users\YourName\Desktop\5th-project_mvp

# .env.example을 .env로 복사
copy .env.example .env

# 메모장으로 .env 파일 열기
notepad .env
```

### Mac/Linux

```bash
# 프로젝트 루트 폴더로 이동
cd ~/Desktop/5th-project_mvp

# .env.example을 .env로 복사
cp .env.example .env

# 텍스트 에디터로 열기
nano .env
# 또는
vi .env
```

---

## ✏️ 2단계: .env 파일 수정

`.env` 파일을 열면 다음과 같은 내용이 보입니다:

```env
# ========================================
# 1. 프로젝트 루트 경로
# ========================================
PROJECT_ROOT=C:\Users\playdata2\Desktop\playdata\Workspace\팀프로젝트5\5th-project_mvp
```

**본인 환경에 맞게 수정하세요:**

### Windows 예시
```env
PROJECT_ROOT=C:\Users\YourName\Desktop\5th-project_mvp
```

### Mac/Linux 예시
```env
PROJECT_ROOT=/Users/YourName/Desktop/5th-project_mvp
```

**💡 Tip: 프로젝트 경로 복사하는 법**

#### Windows
1. 파일 탐색기에서 `5th-project_mvp` 폴더 열기
2. 주소창 클릭 → 경로 복사
3. `.env` 파일에 붙여넣기

#### Mac
1. Finder에서 `5th-project_mvp` 폴더 우클릭
2. "Option" 키 누른 상태에서 "경로 이름 복사"
3. `.env` 파일에 붙여넣기

#### Linux
```bash
pwd  # 현재 경로 출력
```

---

## ✅ 3단계: 설정 확인

설정이 제대로 되었는지 확인:

```bash
# config.py 직접 실행
python config.py
```

**올바른 출력 예시:**
```
============================================================
프로젝트 환경 설정
============================================================
프로젝트 루트: C:\Users\YourName\Desktop\5th-project_mvp
데이터 파일: hint-system\data\problems_multi_solution.json
크롤러 출력: crawler\data\raw
평가 결과: hint-system\evaluation\results
로그 파일: logs\app.log
============================================================
```

---

## 🛠️ 4단계: 가상환경 설정

### Windows

```powershell
# 가상환경 생성
python -m venv venv

# 가상환경 활성화
venv\Scripts\activate

# 프롬프트가 (venv)로 시작하면 성공!
```

### Mac/Linux

```bash
# 가상환경 생성
python3 -m venv venv

# 가상환경 활성화
source venv/bin/activate

# 프롬프트가 (venv)로 시작하면 성공!
```

---

## 📦 5단계: 패키지 설치

### 힌트 시스템 패키지 설치

```bash
cd hint-system
pip install -r requirements.txt
```

### 크롤러 패키지 설치 (크롤링 필요한 경우)

```bash
cd crawler/crawlers
pip install -r requirements.txt
```

---

## 🚀 6단계: 실행

### 힌트 시스템 실행

```bash
cd hint-system
python app.py
```

브라우저에서 `http://localhost:7860` 접속

### 크롤러 실행 (선택사항)

```bash
cd crawler/crawlers
python crawl_all_hybrid.py
```

---

## ❓ 문제 해결 (Troubleshooting)

### 문제 1: `ModuleNotFoundError: No module named 'dotenv'`

**원인:** `python-dotenv` 패키지가 설치되지 않음

**해결:**
```bash
pip install python-dotenv
```

### 문제 2: `FileNotFoundError: [Errno 2] No such file or directory: 'data/...'`

**원인:** `.env` 파일의 경로가 잘못됨

**해결:**
1. `.env` 파일에서 `PROJECT_ROOT` 경로 확인
2. 경로에 역슬래시(`\`)가 있으면 이스케이프 처리:
   - 잘못된 예: `C:\Users\name` (X)
   - 올바른 예: `C:\\Users\\name` 또는 `C:/Users/name` (O)

### 문제 3: `.env` 파일이 보이지 않음

**원인:** 숨김 파일로 설정됨

**해결 (Windows):**
1. 파일 탐색기 → "보기" 탭
2. "숨긴 항목" 체크박스 활성화

**해결 (Mac):**
```bash
# Finder에서 숨김 파일 표시
Cmd + Shift + .
```

**해결 (Linux):**
```bash
# 숨김 파일 보기
ls -la
```

### 문제 4: 가상환경이 활성화되지 않음

**해결 (Windows PowerShell 권한 오류):**
```powershell
# PowerShell 관리자 권한으로 실행 후:
Set-ExecutionPolicy RemoteSigned

# 다시 시도:
venv\Scripts\activate
```

---

## 📋 체크리스트

설정이 완료되었는지 확인하세요:

- [ ] `.env` 파일 생성 완료
- [ ] `.env` 파일에 본인 경로 입력 완료
- [ ] `python config.py` 실행 시 올바른 경로 출력
- [ ] 가상환경 생성 및 활성화 완료
- [ ] `pip install -r requirements.txt` 실행 완료
- [ ] `python app.py` 실행 시 에러 없음

---

## 🎯 다음 단계

모든 설정이 완료되었다면:

1. [README.md](README.md) - 프로젝트 전체 개요
2. [hint-system/README.md](hint-system/README.md) - 힌트 시스템 상세 가이드
3. [crawler/README.md](crawler/README.md) - 크롤러 사용법

---

## 💬 문의

설정 중 문제가 발생하면 팀원에게 문의하세요.
