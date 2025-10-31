# solved.ac API 테스트 파일

크롤러 개발 과정에서 사용한 API 테스트 파일들입니다.

## 📄 파일 목록

### 1. test_solved_ac_api.py
**목적:** solved.ac API 기능 확인

**테스트 내용:**
- 단일 문제 정보 가져오기
- 여러 문제 정보 한 번에 가져오기
- API 응답 구조 확인
- 태그 정보 확인

**실행 방법:**
```bash
python test_solved_ac_api.py
```

**주요 발견:**
- solved.ac는 문제의 난이도(level)와 태그(tags) 제공
- 단계별(step) 정보는 백준에서만 제공 → 하이브리드 크롤링 필요

**API 엔드포인트:**
- 단일 문제: `https://solved.ac/api/v3/problem/show?problemId={id}`
- 다중 문제: `https://solved.ac/api/v3/problem/lookup?problemIds={ids}`

---

### 2. check_tags.py
**목적:** 백준 웹사이트에서 태그 크롤링 가능 여부 확인

**테스트 내용:**
- 백준 문제 페이지에서 태그 정보 크롤링 시도
- BeautifulSoup을 사용한 HTML 파싱

**실행 방법:**
```bash
python check_tags.py
```

**주요 발견:**
- 백준은 비로그인 사용자에게 태그를 보여주지 않음
- 로그인 쿠키 필요 or solved.ac API 사용 권장
- **결론:** solved.ac API가 더 안정적이고 공식적인 방법

---

## 🔧 크롤러 설계 결정

위 테스트 결과를 바탕으로 **하이브리드 크롤러**를 개발:

### 데이터 소스 분담
| 정보 | 출처 | 방법 |
|------|------|------|
| 단계별 문제 목록 | 백준 | HTML 크롤링 |
| 문제 설명, 예제 | 백준 | HTML 크롤링 |
| 난이도(Level) | solved.ac | API |
| 태그(분류) | solved.ac | API |

### 최종 크롤러: baekjoon_hybrid_crawler.py
- 백준에서 단계별 문제 리스트 수집
- 각 문제 상세 정보 크롤링
- solved.ac API로 난이도/태그 추가
- JSON 형식으로 통합 저장

---

## 📚 solved.ac API 문서

공식 문서: https://solvedac.github.io/unofficial-documentation/

**주요 API:**
- `GET /problem/show` - 단일 문제 정보
- `GET /problem/lookup` - 다중 문제 정보 (최대 100개)
- Rate Limit: 300 requests/min (비로그인)

**인증:**
- 선택사항 (비로그인도 가능)
- API 키 발급 시 더 높은 Rate Limit
- 설정 방법: `.env` 파일에 `SOLVED_AC_API_KEY` 추가

---

## 🚀 실전 사용 예시

```python
from baekjoon_hybrid_crawler import BaekjoonHybridCrawler

# 크롤러 초기화
crawler = BaekjoonHybridCrawler()

# Step 1~3 크롤링
problems = crawler.crawl_by_steps(start_step=1, end_step=3)

# 결과 확인
for prob in problems:
    print(f"#{prob['problem_id']}: {prob['title']}")
    print(f"  Level: {prob['level']}, Tags: {prob['tags']}")
```

---

## ⚠️ 주의사항

1. **Rate Limiting**
   - solved.ac API: 300 req/min
   - 크롤러는 자동으로 1초 딜레이 적용

2. **에러 처리**
   - API 실패 시 빈 태그로 저장
   - 백준 크롤링 실패 시 해당 문제 스킵

3. **데이터 정확성**
   - solved.ac 데이터는 커뮤니티 기반
   - 일부 문제는 태그가 없을 수 있음

---

**작성일:** 2025-01-30
**테스트 환경:** Python 3.10+
