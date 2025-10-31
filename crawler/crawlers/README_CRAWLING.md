# 백준 문제 크롤링 가이드

## 📌 개요

백준 단계별 문제를 크롤링하여 JSON 파일로 저장하는 하이브리드 크롤러입니다.

### 데이터 소스
- **백준 (acmicpc.net)**: 단계, 문제 설명, 예제 입출력
- **solved.ac API**: 태그(분류), 난이도, 통계

### 생성되는 JSON 구조
```json
{
  "problem_id": "2557",
  "title": "Hello World",
  "level": 1,
  "tags": ["구현"],
  "description": "Hello World!를 출력하시오.",
  "input_description": "없음",
  "output_description": "Hello World!를 출력하시오.",
  "examples": [{"input": "", "output": "Hello World!"}],
  "step": 1,
  "step_title": "입출력과 사칙연산",
  "accepted_user_count": 391041,
  "average_tries": 2.5273,
  "url": "https://www.acmicpc.net/problem/2557"
}
```

---

## 🚀 크롤링 실행 방법

### 방법 1: 대화형 모드 (추천)

```bash
# 1. 크롤러 디렉토리로 이동
cd C:\Users\playdata2\Desktop\playdata\Workspace\팀프로젝트5\5th-project_mvp\data-pipeline\crawlers

# 2. 크롤러 실행
..\..\..\..\..\.venv\Scripts\python.exe baekjoon_hybrid_crawler.py

# 3. 옵션 선택
# 1. Step 1~3 (빠른 테스트)
# 2. Step 1~10 (추천)
# 3. Step 1~68 (전체)
# 4. 직접 입력
```

### 방법 2: 전체 크롤링 스크립트

```bash
# 1. 크롤러 디렉토리로 이동
cd C:\Users\playdata2\Desktop\playdata\Workspace\팀프로젝트5\5th-project_mvp\data-pipeline\crawlers

# 2. 전체 크롤링 실행
..\..\..\..\..\.venv\Scripts\python.exe crawl_all_hybrid.py

# 3. 확인 (y 입력)
# 계속하시겠습니까? (y/n): y
```

### 방법 3: Python 코드로 직접 실행

```python
from baekjoon_hybrid_crawler import BaekjoonHybridCrawler

# 크롤러 생성
crawler = BaekjoonHybridCrawler(output_dir="../data/raw")

# Step 1~10 크롤링
problems = crawler.crawl_by_steps(
    start_step=1,
    end_step=10,
    delay=1.0  # 요청 간 대기 시간 (초)
)

print(f"총 {len(problems)}개 문제 수집 완료!")
```

---

## ⏱️ 예상 소요 시간

| 범위 | 예상 문제 수 | 예상 시간 |
|------|-------------|----------|
| Step 1~3 | ~30개 | 1분 |
| Step 1~10 | ~100개 | 3-5분 |
| Step 1~68 (전체) | ~1000개 | 30-60분 |

**주의:** 백준 서버 부하 방지를 위해 각 요청 사이에 1초씩 대기합니다.

---

## 📁 출력 파일

### 저장 위치
```
5th-project_mvp/data-pipeline/data/raw/
```

### 파일명 형식
```
problems_hybrid_step_{시작단계}_to_{종료단계}.json
```

### 예시
```
problems_hybrid_step_1_to_3.json    # Step 1~3
problems_hybrid_step_1_to_10.json   # Step 1~10
problems_hybrid_step_1_to_68.json   # 전체
```

---

## 📊 크롤링 결과 확인

### 파일 확인
```bash
# JSON 파일 존재 확인
dir ..\data\raw\problems_hybrid_*.json
```

### Python으로 확인
```python
import json

# JSON 파일 읽기
with open('../data/raw/problems_hybrid_step_1_to_68.json', encoding='utf-8') as f:
    problems = json.load(f)

print(f"총 {len(problems)}개 문제")

# 단계별 통계
from collections import Counter
steps = Counter(p['step'] for p in problems)
for step, count in sorted(steps.items()):
    print(f"  Step {step}: {count}개")

# 태그 통계
total_tags = sum(len(p['tags']) for p in problems)
print(f"\n총 태그 수: {total_tags}개")
print(f"평균 태그 수: {total_tags/len(problems):.2f}개/문제")
```

---

## ⚠️ 주의사항

### 1. 일부 단계는 삭제됨
- **Step 5, 12 등**은 백준에서 삭제되어 404 에러 발생
- 크롤러가 자동으로 건너뜁니다 (에러 아님)

### 2. 네트워크 오류
- 간혹 네트워크 오류로 일부 문제 수집 실패 가능
- `[FAIL]` 표시된 문제는 재시도 권장

### 3. 서버 부하
- 대기 시간(delay) 1초 미만으로 설정 금지
- 백준 서버에 과도한 부하를 주지 않도록 주의

---

## 🔧 문제 해결

### Q1: `ModuleNotFoundError: No module named 'requests'`
```bash
# 필수 패키지 설치
..\..\..\..\..\.venv\Scripts\pip.exe install requests beautifulsoup4
```

### Q2: 크롤링이 중간에 멈춤
- 네트워크 연결 확인
- 백준 사이트 접속 가능 여부 확인
- 중단된 단계부터 재시작 가능

### Q3: 태그가 비어있음
- solved.ac API 응답 확인 필요
- 일부 오래된 문제는 태그가 없을 수 있음

---

## 📞 추가 도움

문제가 발생하면:
1. `baekjoon_hybrid_crawler.py` 코드 확인
2. 에러 메시지 확인
3. 백준/solved.ac 사이트 접속 여부 확인
