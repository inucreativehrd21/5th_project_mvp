# 프롬프트 수정 V3 - 함수명 언급 완전 차단

## 문제 상황

### 여전히 함수명이 나옴
```
📌 Qwen2.5-Coder-1.5B
"count_repaint 함수를 어떻게 만들어야 할까"
```

**문제:**
- V2에서 프롬프트에 **나쁜 예시로 "count_repaint 함수를..."을 보여줌**
- 모델이 나쁜 예시를 따라함
- solution_code를 보여줘서 모델이 함수명을 직접 볼 수 있었음

---

## 핵심 수정 사항

### 1. 프롬프트에서 나쁜 예시 완전 제거 ([app.py:314-343](app.py#L314-L343))

**Before (V2):**
```python
[나쁜 질문 예시]
- "count_repaint 함수를 어떻게 만들어야 할까?" (함수명 언급)
- "위 코드처럼 하려면..." (정답 코드 언급)
```
→ 모델이 이걸 따라함!

**After (V3):**
```python
좋은 질문 예시:
- 함수가 필요한 경우: "같은 계산을 여러 번 재사용하려면 어떤 구조가 필요할까?"
- 반복문이 필요한 경우: "모든 가능한 경우를 검사하려면 어떤 반복 구조가 필요할까?"
- 조건문이 필요한 경우: "여러 조건에 따라 다르게 처리하려면 어떤 구문이 필요할까?"
- 최솟값이 필요한 경우: "여러 값 중 가장 작은 값을 추적하려면 어떤 방법을 쓸까?"
```
→ **나쁜 예시 제거, 좋은 예시만 제공!**

### 2. solution_code 제거

**Before:**
```python
[참고: 완성된 코드는 이런 구조를 가집니다]
{solution_code}
```
→ 모델이 solution_code에서 함수명을 직접 볼 수 있음

**After:**
```python
학생 코드:
{user_code}

분석:
- 완료한 것: {completed_desc}
- 다음 할 일: {next_step_goal}
```
→ **solution_code 완전 제거!**

### 3. 함수명/변수명 자동 필터링 ([model_inference.py:246-257](models/model_inference.py#L246-L257))

**신규 필터:**
```python
# snake_case 패턴 감지 (count_repaint, min_count 등)
function_name_pattern = r'\b[a-z]+_[a-z]+\b|[a-z]+[A-Z][a-z]+'
if re.search(function_name_pattern, sentence):
    # 일반 단어 제외 (input, print, range 등)
    common_words = ['input', 'print', 'range', 'split', 'append']
    suspicious_names = re.findall(function_name_pattern, sentence)
    if any(name not in common_words for name in suspicious_names):
        print(f"[DEBUG] 함수명/변수명 언급 필터링: {sentence[:50]}...")
        continue
```

**작동 방식:**
- ✅ `count_repaint` → 필터링 (snake_case)
- ✅ `min_count` → 필터링 (snake_case)
- ✅ `countRepaint` → 필터링 (camelCase)
- ⭕ `input()` → 통과 (일반 단어)
- ⭕ `print()` → 통과 (일반 단어)

---

## 최종 프롬프트 구조

```python
학생 코드:
{user_code}

분석:
- 완료한 것: {completed_desc}
- 다음 할 일: {next_step_goal}

임무:
"{next_step_goal}"를 학생이 스스로 생각하게 만드는 소크라테스식 질문을 한국어로 만드세요.

중요 규칙:
1. 구체적인 이름(함수명, 변수명)을 절대 쓰지 마세요
2. 일반적인 개념만 사용: 함수, 반복문, 조건문, 변수 등
3. 형식: "~하려면 어떤 방법/구문이 필요할까?"

좋은 질문 예시:
- 함수가 필요한 경우: "같은 계산을 여러 번 재사용하려면 어떤 구조가 필요할까?"
- 반복문이 필요한 경우: "모든 가능한 경우를 검사하려면 어떤 반복 구조가 필요할까?"
- 조건문이 필요한 경우: "여러 조건에 따라 다르게 처리하려면 어떤 구문이 필요할까?"
- 최솟값이 필요한 경우: "여러 값 중 가장 작은 값을 추적하려면 어떤 방법을 쓸까?"

반말로 한 문장 질문:
```

**핵심 변경:**
- ❌ solution_code 제거 → 모델이 정답 코드를 볼 수 없음
- ❌ 나쁜 예시 제거 → 모델이 잘못된 패턴을 따라하지 않음
- ✅ 좋은 예시만 제공 → 올바른 패턴만 학습

---

## 3단계 방어선

### 1단계: 프롬프트 (예방)
- solution_code 노출 안 함
- 나쁜 예시 제거
- 좋은 예시만 제공

### 2단계: 모델 생성 (가이드)
- 명확한 규칙 제시
- "함수명/변수명 절대 쓰지 마세요"
- 형식 가이드 제공

### 3단계: 필터링 (차단)
- snake_case 패턴 자동 감지
- camelCase 패턴 자동 감지
- 일반 단어는 예외 처리

---

## 기대 결과

### 체스판 문제 (입력만 완료)

**다음 단계:** 함수 정의

**이제 받을 수 있는 힌트:**
- ✅ "같은 계산을 여러 번 재사용하려면 어떤 구조가 필요할까?"
- ✅ "8x8 영역을 검사하는 작업을 묶어서 만들려면 어떤 키워드가 필요할까?"
- ✅ "반복적으로 호출할 작업을 정의하려면 어떤 방법을 써야 할까?"

**이제 완전히 차단되는 힌트:**
- ❌ "count_repaint 함수를 어떻게 만들어야 할까?" → **snake_case 필터 작동**
- ❌ "min_count 변수를 어떻게 초기화할까?" → **snake_case 필터 작동**
- ❌ "countRepaint를 만들려면?" → **camelCase 필터 작동**

---

## 변경 파일

1. **app.py** (line 314-343)
   - solution_code 제거
   - 나쁜 예시 제거
   - 좋은 예시 확장 (4가지)

2. **models/model_inference.py** (line 246-257)
   - snake_case 필터 추가
   - camelCase 필터 추가
   - 일반 단어 예외 처리

---

## 테스트

```bash
# 1. 앱 실행
python app.py

# 2. 체스판 문제 선택

# 3. 입력만 작성:
N, M = map(int, input().split())
board = []
for _ in range(N):
    board.append(input())

# 4. 힌트 요청

# 5. 확인:
# - "count_repaint" 같은 함수명 언급이 없는지
# - "~하려면 어떤 방법/구문이 필요할까?" 형식인지
```

---

## 변경 이력

- **V1 (2025-01-30):** 코드 구조 분석 + 프롬프트 컨텍스트 강화
- **V2 (2025-01-30):** 정답 코드 언급 금지, 직접 답 요구 차단
- **V3 (2025-01-30):** 나쁜 예시 제거, solution_code 제거, snake_case 자동 필터링 ✅

---

## 날짜
2025-01-30 (V3 최종)
