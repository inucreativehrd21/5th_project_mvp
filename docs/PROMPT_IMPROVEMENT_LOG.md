# 프롬프트 개선 로그

## 문제 상황 (Before)

### 체스판 문제 (#1018) 테스트 케이스

**정답 코드:**
```python
N, M = map(int, input().split())
board = []
for _ in range(N):
    board.append(input())

def count_repaint(x, y, first):
    count = 0
    for i in range(8):
        for j in range(8):
            if (i + j) % 2 == 0:
                if board[x+i][y+j] != first:
                    count += 1
            else:
                if board[x+i][y+j] == first:
                    count += 1
    return count

min_count = 64
for i in range(N - 7):
    for j in range(M - 7):
        count_W = count_repaint(i, j, 'W')
        count_B = count_repaint(i, j, 'B')
        min_count = min(min_count, count_W, count_B)

print(min_count)
```

**사용자 코드 (입력만 완료):**
```python
N, M = map(int, input().split())
board = []
for _ in range(N):
    board.append(input())
```

**받은 잘못된 힌트:**
1. "결과 출력을 하려면 어떤 방법/구문이 필요합니다" ← 너무 앞서감 (함수도 없는데 출력 얘기)
2. "(모델이 한국어 힌트를 생성하지 못했습니다)" ← 완전 실패

### 근본 원인

**기존 프롬프트 (~90 토큰):**
```
정답:
{solution_code}

학생 작성:
{user_code}

다음 할 일: {next_hint}

학생이 다음 단계를 스스로 떠올리게 만드는 한국어 질문 하나를 만드세요.

질문 형식: "[다음 할 일]을 하려면 어떤 방법/구문이 필요할까요?"

질문:
```

**문제점:**
1. **컨텍스트 부족:** 학생이 무엇을 완료했는지, 무엇이 남았는지 명확하지 않음
2. **next_hint 계산 오류:** logic_steps의 단순 패턴 매칭으로는 복잡한 코드 분석 실패
3. **너무 짧은 지시:** 1.5B 소형 모델에게 부족한 가이드

---

## 개선 사항 (After)

### 1. 코드 구조 분석 추가

**새 메서드: `_analyze_code_structure()`**
```python
def _analyze_code_structure(self, code: str) -> Dict[str, bool]:
    return {
        'has_function': 'def ' in code,
        'has_for_loop': 'for ' in code,
        'has_while_loop': 'while ' in code,
        'has_if': 'if ' in code,
        'has_elif': 'elif ' in code,
        'has_print': 'print(' in code,
        'has_return': 'return ' in code,
        'has_input': 'input()' in code,
        'line_count': len([l for l in code.split('\n') if l.strip()])
    }
```

**새 메서드: `_describe_code_diff()`**
- 정답 코드와 사용자 코드의 구조적 차이를 분석
- "완료한 것", "아직 작성하지 않은 것"을 명확히 파악

**체스판 문제 분석 결과:**
```
완료: 입력 받기, 반복문
누락: 함수 정의, 조건문 작성, 결과 출력
다음 단계: 함수 정의  ← 정확!
```

### 2. 프롬프트 템플릿 개선 (~180 토큰)

**개선된 프롬프트:**
```
정답 코드:
{solution_code}

학생 코드:
{user_code}

[분석]
학생이 완료한 것: {completed_desc}
아직 작성하지 않은 것: {missing_desc}
다음 단계: {next_step_goal}

[중요 규칙]
1. 학생이 다음 단계를 스스로 생각하게 유도하는 질문 만들기
2. 이미 작성한 코드는 언급하지 말 것
3. "[목적]을 하려면 어떤 [방법/구문]이 필요할까?" 형식 사용
4. 한국어 반말로 한 문장만
5. 구체적이고 명확하게 (추상적인 표현 금지)

질문:
```

**개선점:**
- ✅ 명확한 컨텍스트 제공 ([분석] 섹션)
- ✅ 구체적인 지시사항 (5가지 규칙)
- ✅ "추상적 표현 금지" 명시

### 3. 필터 강화 (models/model_inference.py)

**기존 필터:**
- 최소 길이: 20자
- 한국어 비율: 30% 이상
- 메타 표현 제거

**추가된 필터:**

#### (1) 최소 길이 상향: 20자 → 30자
```python
if len(sentence) < 30:
    print(f"[DEBUG] 너무 짧은 문장 필터링: {sentence}")
    continue
```

#### (2) 추상적 표현만 있는 힌트 거부 (NEW!)
```python
abstract_only = [
    '결과 출력', '코드 작성', '로직 구현', '알고리즘 작성',
    '문제 해결', '방법 찾기', '구조 만들기'
]

# 구체적인 키워드 체크
has_concrete = any(keyword in sentence for keyword in [
    '함수', '반복문', 'for', 'while', 'def', '조건문', 'if',
    '변수', '리스트', '배열', '입력', 'input', '출력', 'print',
    'append', 'range', 'len', 'max', 'min', 'sum', 'sorted'
])

# 추상적 표현만 있고 구체적 키워드가 없으면 거부
if any(abstract in sentence for abstract in abstract_only) and not has_concrete:
    print(f"[DEBUG] 추상적 표현만 있는 힌트 필터링: {sentence[:50]}...")
    continue
```

**효과:**
- ❌ "결과 출력을 하려면..." ← **거부됨** (추상적 + 구체적 키워드 없음)
- ✅ "결과 출력을 위해 print 함수를 사용하려면..." ← 통과 (print 키워드 있음)

---

## 기대 효과

### 체스판 문제에서 기대되는 힌트:

**좋은 힌트 예시:**
- "8x8 체스판을 검사하는 작업을 재사용하려면 어떤 구문으로 묶어야 할까?"
- "특정 영역의 칠해야 할 개수를 계산하는 함수를 만들려면 어떤 키워드가 필요할까?"
- "같은 작업을 여러 번 호출하려면 어떤 구조로 코드를 작성해야 할까?"

**이제 필터링되는 나쁜 힌트:**
- ❌ "결과 출력을 하려면..." (추상적 표현 + 너무 앞서감)
- ❌ 20자 미만의 짧은 힌트
- ❌ 한국어 30% 미만의 영어 출력

---

## 변경 파일

1. **app.py**
   - `_analyze_code_structure()` 추가 (line 321-333)
   - `_describe_code_diff()` 추가 (line 335-369)
   - `_create_analysis_prompt()` 프롬프트 개선 (line 305-334)

2. **models/model_inference.py**
   - 최소 길이 20 → 30자 상향 (line 262)
   - 추상적 표현 필터 추가 (line 266-279)

---

## 테스트 방법

```bash
# 앱 실행
python app.py

# 체스판 문제 (#1018) 선택
# 사용자 코드 입력:
N, M = map(int, input().split())
board = []
for _ in range(N):
    board.append(input())

# 힌트 요청
# 기대: "함수 정의"와 관련된 구체적인 힌트
```

---

## 비교 요약

| 항목 | Before | After |
|------|--------|-------|
| 프롬프트 길이 | ~90 토큰 | ~180 토큰 |
| 컨텍스트 | "다음 할 일" 텍스트만 | 완료/누락 분석 제공 |
| 코드 분석 | logic_steps 패턴 매칭 | 구조적 차이 분석 |
| 최소 힌트 길이 | 20자 | 30자 |
| 추상적 표현 필터 | 없음 | 추가됨 |
| 체스판 문제 정확도 | "결과 출력" (잘못됨) | "함수 정의" (정확) |

---

## 날짜

2025-01-30
