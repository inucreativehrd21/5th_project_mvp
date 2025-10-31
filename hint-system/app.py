"""
단일 화면 코딩 힌트 평가 시스템
"""
import gradio as gr
import json
import os
import sys
from datetime import datetime
from typing import List, Dict, Optional
from pathlib import Path

# 프로젝트 루트를 sys.path에 추가 (config.py import를 위해)
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config import Config
from models.model_inference import ModelManager


class HintEvaluationApp:
    """힌트 평가 애플리케이션"""

    def __init__(self, data_path: str, auto_setup_models: bool = True):
        self.data_path = data_path
        self.problems = self.load_problems()
        self.model_manager = ModelManager(sequential_load=True)  # 순차 로드 활성화
        self.current_problem = None
        self.current_hints = {}  # 모델별 힌트 저장
        self.evaluation_results = []

        # 평가 결과 저장 경로 (config에서 자동 설정)
        self.results_dir = Config.EVALUATION_RESULTS_DIR
        os.makedirs(self.results_dir, exist_ok=True)

        # 자동으로 기본 모델 설정
        if auto_setup_models:
            self.setup_default_models()

    def setup_default_models(self):
        """기본 모델 자동 설정 (vLLM 우선, HuggingFace 폴백)"""
        print("=" * 60)
        print("🚀 모델 자동 설정 중...")
        print("=" * 60)

        # vLLM 서버 연결 시도 (빠른 추론을 위해 우선 시도)
        vllm_url = Config.VLLM_SERVER_URL
        vllm_connected = False

        if vllm_url:
            print(f"\n⚡ vLLM 서버 연결 시도: {vllm_url}")
            try:
                # vLLM 서버에 연결된 모델 자동 감지 및 추가
                self.model_manager.add_vllm_model(
                    model_name="vLLM-Server",
                    model_path="auto",  # 서버에서 자동으로 감지
                    base_url=vllm_url
                )
                print(f"✅ vLLM 서버 연결 성공! (15-24배 빠른 추론 속도)")
                vllm_connected = True
            except Exception as e:
                print(f"⚠️  vLLM 서버 연결 실패: {e}")
                print(f"    HuggingFace 모델로 폴백합니다...")

        # vLLM 연결 실패 시 또는 추가 모델이 필요한 경우
        if not vllm_connected:
            print(f"\n📦 HuggingFace 모델 로딩 중 (느릴 수 있습니다)...")
            print(f"    💡 Tip: vLLM 서버를 사용하면 15-24배 빠릅니다!")
            print(f"    실행: python vllm_server.py --model Qwen/Qwen2.5-Coder-7B-Instruct\n")

            # 가벼운 모델 5개 (로컬 HuggingFace Transformers)
            default_models = [
                {
                    "name": "Qwen2.5-Coder-1.5B",
                    "path": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
                    "quantize": False,
                    "size": "1.5B"
                },
                {
                    "name": "DeepSeek-Coder-1.3B",
                    "path": "deepseek-ai/deepseek-coder-1.3b-instruct",
                    "quantize": False,
                    "size": "1.3B"
                },
                {
                    "name": "TinyLlama-1.1B",
                    "path": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                    "quantize": False,
                    "size": "1.1B"
                },
                {
                    "name": "Phi-2-2.7B (4-bit)",
                    "path": "microsoft/phi-2",
                    "quantize": True,
                    "size": "2.7B"
                },
                {
                    "name": "CodeLlama-7B (4-bit)",
                    "path": "codellama/CodeLlama-7b-Instruct-hf",
                    "quantize": True,
                    "size": "7B"
                }
            ]

            for model_info in default_models:
                try:
                    print(f"\n📦 {model_info['name']} 설정 중...")
                    if model_info.get('quantize'):
                        print(f"  → 4-bit 양자화 사용 (메모리 1/4 절약)")
                    self.model_manager.add_huggingface_model(
                        model_info['name'],
                        model_info['path'],
                        use_quantization=model_info.get('quantize', False)
                    )
                    print(f"✅ {model_info['name']} 추가 완료!")
                except Exception as e:
                    print(f"⚠️  {model_info['name']} 추가 실패: {e}")

        print("\n" + "=" * 60)
        print(f"✅ 총 {len(self.model_manager.get_available_models())}개 모델 준비 완료!")
        if vllm_connected:
            print(f"⚡ vLLM 고속 추론 활성화됨!")
        print("=" * 60)

    def load_problems(self) -> List[Dict]:
        """문제 데이터 로드 (multi-solution 구조)"""
        with open(self.data_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def get_problem_list(self) -> List[str]:
        """문제 목록 (드롭다운용)"""
        return [
            f"#{p['problem_id']} - {p['title']} (Level {p['level']})"
            for p in self.problems
        ]

    def load_problem(self, problem_selection: str):
        """선택된 문제 로드"""
        if not problem_selection:
            return self._empty_state()

        try:
            # 문제 ID 추출
            problem_id = problem_selection.split('#')[1].split(' -')[0].strip()

            # 문제 찾기
            self.current_problem = None
            for p in self.problems:
                if str(p['problem_id']) == str(problem_id):
                    self.current_problem = p
                    break

            if not self.current_problem:
                return "❌ 문제를 찾을 수 없습니다.", gr.update(value=""), *self._empty_hints()

            # 문제 정보 포맷팅
            problem_md = self._format_problem_display()

            # 힌트 영역 초기화
            return problem_md, gr.update(value="# 여기에 코드를 작성하세요\n"), *self._empty_hints()

        except Exception as e:
            print(f"[ERROR] 문제 로드 실패: {e}")
            import traceback
            traceback.print_exc()
            return f"❌ 오류: {str(e)}", gr.update(value=""), *self._empty_hints()

    def _format_problem_display(self) -> str:
        """문제 표시 포맷"""
        p = self.current_problem
        md = f"""# {p['title']}

**난이도:** Level {p['level']} | **태그:** {', '.join(p['tags'])} | **문제 링크:** [{p['problem_id']}]({p['url']})

---

## 📋 문제 설명
{p['description']}

## 📥 입력
{p['input_description']}

## 📤 출력
{p['output_description']}

## 💡 예제
"""
        for i, example in enumerate(p['examples'], 1):
            input_txt = example.get('input', '') if example.get('input') else '(없음)'
            output_txt = example.get('output', '') if example.get('output') else '(없음)'
            md += f"\n**예제 {i}**\n```\n입력: {input_txt}\n출력: {output_txt}\n```\n"

        # 정답 코드 추가 (여러 solution 지원)
        solutions = p.get('solutions', [])
        if solutions:
            md += "\n\n---\n\n## ✅ 정답 코드 (참고용)\n\n"

            if len(solutions) == 1:
                # 단일 풀이
                sol = solutions[0]
                md += f"```python\n{sol.get('solution_code', '# 정답 코드가 없습니다.')}\n```\n"

                # Logic 단계
                if sol.get('logic_steps'):
                    md += "\n### 🎯 Logic 단계 (문제 해결 과정)\n\n"
                    for i, step in enumerate(sol['logic_steps'], 1):
                        md += f"{i}. **{step.get('goal', '')}**\n"
                        if step.get('socratic_hint'):
                            md += f"   - 힌트 예시: _{step['socratic_hint']}_\n"
                        md += "\n"
            else:
                # 다중 풀이
                md += f"**이 문제는 {len(solutions)}가지 풀이 방법이 있습니다. 원하는 방식으로 풀어보세요!**\n\n"

                for sol in solutions:
                    sol_name = sol.get('solution_name', f"풀이 {sol.get('solution_id', '')}")
                    md += f"### {sol_name}\n\n"
                    md += f"```python\n{sol.get('solution_code', '')}\n```\n\n"

                    # Logic 단계 (접기)
                    if sol.get('logic_steps'):
                        md += "<details>\n<summary>🎯 Logic 단계 보기</summary>\n\n"
                        for i, step in enumerate(sol['logic_steps'], 1):
                            md += f"{i}. **{step.get('goal', '')}**\n"
                            if step.get('socratic_hint'):
                                md += f"   - 힌트 예시: _{step['socratic_hint']}_\n"
                            md += "\n"
                        md += "</details>\n\n"

            md += """
**💡 힌트 평가 팁:**
1. 위 정답 코드를 참고하여 **일부러 틀린 코드**를 작성하세요
2. 예: 반복 횟수 틀리기, 변수명 잘못 쓰기, 로직 누락 등
3. 모델이 어떻게 힌트를 주는지 평가하세요!
"""

        return md

    def request_hints(self, user_code: str, temperature: float, *selected_models):
        """선택된 모델들에게 힌트 요청 (temperature 조절 가능)"""
        if not self.current_problem:
            return "❌ 먼저 문제를 선택해주세요.", *self._empty_hints()

        if not user_code.strip():
            return "❌ 코드를 입력해주세요.", *self._empty_hints()

        # 선택된 모델 필터링
        available_models = self.model_manager.get_available_models()
        selected = [m for i, m in enumerate(available_models) if i < len(selected_models) and selected_models[i]]

        if not selected:
            return "❌ 최소 1개 모델을 선택해주세요.", *self._empty_hints()

        # 프롬프트 생성 (간결한 Zero-shot)
        prompt = self._create_analysis_prompt(user_code)

        # 선택된 모델로부터만 힌트 생성 (temperature 전달)
        print(f"\n🤖 선택된 {len(selected)}개 모델에게 힌트 요청 중... (temperature={temperature})")
        results = self.model_manager.generate_hints_from_selected(prompt, selected, temperature=temperature)
        self.current_hints = results

        # 각 모델별로 반환할 컴포넌트 생성
        hints = []
        ratings = []
        available_models = self.model_manager.get_available_models()

        status_msg = f"✅ {len(available_models)}개 모델의 힌트 생성 완료!"

        for model_name in available_models:
            result = results.get(model_name, {})

            if result.get('error'):
                hint_display = f"❌ **에러:** {result['error']}"
            else:
                hint_display = result.get('hint', '(빈 응답)')

            hints.append(gr.update(value=hint_display, visible=True))
            ratings.append(gr.update(value=3, visible=True))

        # 남은 슬롯 채우기 (최대 5개 모델 지원)
        while len(hints) < 5:
            hints.append(gr.update(value="", visible=False))
            ratings.append(gr.update(value=3, visible=False))

        # hints 먼저, ratings 나중 (UI 순서와 일치)
        return status_msg, *(hints[:5] + ratings[:5])

    def _create_analysis_prompt(self, user_code: str) -> str:
        """Logic 기반 힌트 생성 (multi-solution 지원)"""
        p = self.current_problem

        # 사용자 코드와 가장 유사한 solution 찾기
        best_solution = self._find_best_matching_solution(user_code, p.get('solutions', []))

        solution_code = best_solution.get('solution_code', '')
        logic_steps = best_solution.get('logic_steps', [])

        # 학생이 완료한 Logic 단계 찾기 (패턴 매칭)
        completed_step = 0
        for i, step in enumerate(logic_steps):
            pattern = step.get('code_pattern', '')
            if pattern:
                key_keywords = self._extract_keywords(pattern)
                if all(kw in user_code for kw in key_keywords[:2]):  # 주요 키워드 2개만 체크
                    completed_step = i + 1

        # 다음에 해야 할 작업들을 큰 그림으로 묶기
        next_hint = ""
        remaining_steps = []

        if completed_step < len(logic_steps):
            # 남은 모든 단계의 목표 수집
            for i in range(completed_step, len(logic_steps)):
                step = logic_steps[i]
                remaining_steps.append(step.get('goal', ''))

            # 조건문/반복문 패턴 감지
            next_pattern = logic_steps[completed_step].get('code_pattern', '')

            if 'if' in next_pattern:
                # 조건문이 나오면 → 모든 조건 분기를 하나로 묶어서 설명
                if 'elif' in next_pattern or any('elif' in logic_steps[i].get('code_pattern', '') for i in range(completed_step, len(logic_steps))):
                    # 여러 분기 존재 → 전체 케이스 나열
                    next_hint = f"여러 경우를 구분하여 처리 ({', '.join(remaining_steps[:4])})"
                else:
                    next_hint = remaining_steps[0] if remaining_steps else ""

            elif 'for' in next_pattern or 'while' in next_pattern:
                # 반복문 → 반복 목적 + 내부 작업 설명
                loop_goal = remaining_steps[0] if remaining_steps else ""
                inner_goals = remaining_steps[1:3] if len(remaining_steps) > 1 else []
                if inner_goals:
                    next_hint = f"{loop_goal}, 그 안에서 {' 및 '.join(inner_goals)}"
                else:
                    next_hint = loop_goal

            else:
                # 일반 단계 → 다음 2-3개 단계 묶기
                next_hint = " → ".join(remaining_steps[:3]) if remaining_steps else ""

        # 코드 구조 차이 분석
        completed_desc, missing_desc = self._describe_code_diff(user_code, solution_code)

        # 다음 단계 결정: missing 중 첫 번째 항목 or logic_steps 기반
        if missing_desc != "없음":
            next_step_goal = missing_desc.split(", ")[0]  # 첫 번째 누락 항목
        else:
            next_step_goal = next_hint if next_hint else "코드 완성"

        # V6 강화된 소크라테스식 프롬프트 - 동기 유발 & 상상력 자극
        prompt = f"""당신은 학생의 호기심을 자극하고 스스로 발견하게 만드는 창의적 멘토입니다.

### 학생의 현재 코드:
```python
{user_code}
```

### 코드 상황 분석:
- 이미 구현한 것: {completed_desc}
- 다음 도전 과제: {next_step_goal}

### 핵심 미션:
학생이 "{next_step_goal}"의 필요성을 **스스로 깨닫고 열망하도록** 만드세요.
직접 답을 주지 말고, 학생의 상상력과 호기심을 폭발시키는 질문을 던지세요.

### 동기 유발 전략 (반드시 적용):

1. **규모 확장 시나리오**
   - 지금은 작동하지만, 데이터가 1000배 늘어나면?
   - 사용자가 100만 명이 되면?
   - 입력이 예상과 완전히 다르면?

2. **실생활 연결**
   - 유튜브는 수백만 영상을 어떻게 관리할까?
   - 게임에서 아이템이 수천 개면 어떻게 처리할까?
   - 넷플릭스는 사용자마다 다른 추천을 어떻게 할까?

3. **불편함 자극**
   - 같은 코드를 100번 복사해야 한다면?
   - 매번 손으로 하나씩 확인해야 한다면?
   - 내일 요구사항이 바뀐다면?

4. **호기심 유발**
   - 왜 프로 개발자들은 항상 이 패턴을 사용할까?
   - 이 방법이 없었던 옛날에는 어떻게 했을까?
   - 더 똑똑한 방법이 있다면 어떻게 보일까?

5. **성취감 예고**
   - 이것만 해결하면 훨씬 강력해질 텐데
   - 한 줄만 바꾸면 모든 걸 자동화할 수 있는데
   - 이 개념을 알면 다른 문제도 쉽게 풀릴 텐데

### 절대 금지 사항:
❌ 함수명, 변수명, 코드 키워드 직접 언급
❌ "for 반복문", "if 조건문" 같은 기술 용어
❌ "~를 사용하세요", "~를 추가하세요" 같은 직접 지시
❌ 정답의 힌트가 되는 구체적 표현
❌ 예시 코드 조각

### 반드시 포함할 요소:
✓ "만약 ~라면?" 형식의 가정
✓ 상상력을 자극하는 구체적 상황
✓ 학생이 "아, 이래서 필요하구나!" 하게 만드는 통찰
✓ 호기심과 도전 욕구를 일깨우는 톤
✓ 한국어 반말 (친근하고 편안하게)

### 출력 형식:
단 1개의 질문만 작성하세요. 설명, 답변, 추가 힌트 일체 금지.
질문은 30-50단어 이내로 간결하면서도 강렬하게.

질문:"""
        return prompt

    def _analyze_code_structure(self, code: str) -> Dict[str, bool]:
        """코드 구조 분석 (함수, 반복문, 조건문 등)"""
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

    def _describe_code_diff(self, user_code: str, solution_code: str) -> tuple:
        """사용자 코드와 정답 코드의 차이 분석"""
        user_struct = self._analyze_code_structure(user_code)
        solution_struct = self._analyze_code_structure(solution_code)

        completed = []
        missing = []

        if user_struct['has_input']:
            completed.append("입력 받기")

        if solution_struct['has_function'] and not user_struct['has_function']:
            missing.append("함수 정의")
        elif user_struct['has_function']:
            completed.append("함수 정의")

        if solution_struct['has_for_loop'] and not user_struct['has_for_loop']:
            missing.append("반복문 작성")
        elif user_struct['has_for_loop']:
            completed.append("반복문")

        if solution_struct['has_if'] and not user_struct['has_if']:
            missing.append("조건문 작성")
        elif user_struct['has_if']:
            completed.append("조건문")

        if solution_struct['has_print'] and not user_struct['has_print']:
            missing.append("결과 출력")
        elif user_struct['has_print']:
            completed.append("출력")

        completed_desc = ", ".join(completed) if completed else "없음"
        missing_desc = ", ".join(missing) if missing else "없음"

        return completed_desc, missing_desc

    def _extract_keywords(self, pattern: str) -> list:
        """코드 패턴에서 핵심 키워드 추출"""
        import re
        # 변수명, 함수명 추출 (우선순위 순)
        keywords = []

        # 1순위: 사용자 정의 변수/함수
        user_defined = re.findall(r'\b[a-z_][a-z0-9_]*\b', pattern)
        keywords.extend([k for k in user_defined if k not in ['if', 'for', 'while', 'elif', 'else', 'and', 'or', 'not', 'in', 'is']])

        # 2순위: 내장 함수
        builtins = re.findall(r'\b(int|input|print|range|append|max|min|len|sum)\b', pattern)
        keywords.extend(builtins)

        # 중복 제거하되 순서 유지
        seen = set()
        result = []
        for k in keywords:
            if k not in seen:
                seen.add(k)
                result.append(k)

        return result[:4]  # 최대 4개

    def _find_best_matching_solution(self, user_code: str, solutions: List[Dict]) -> Dict:
        """사용자 코드와 가장 유사한 solution 찾기"""
        if not solutions:
            return {}

        # 사용자 코드의 주요 패턴 추출
        user_patterns = {
            'uses_split': '.split()' in user_code or 'split(' in user_code,
            'multiple_inputs': user_code.count('input()') > 1,
            'uses_map': 'map(' in user_code,
            'uses_list_comp': '[' in user_code and 'for' in user_code and ']' in user_code,
            'uses_for_loop': 'for ' in user_code,
            'uses_while_loop': 'while ' in user_code,
            'uses_if': 'if ' in user_code,
            'uses_elif': 'elif ' in user_code,
            'uses_function_def': 'def ' in user_code,
        }

        best_match = None
        max_score = -1

        for solution in solutions:
            sol_code = solution.get('solution_code', '')

            # 각 solution의 패턴 추출
            sol_patterns = {
                'uses_split': '.split()' in sol_code or 'split(' in sol_code,
                'multiple_inputs': sol_code.count('input()') > 1,
                'uses_map': 'map(' in sol_code,
                'uses_list_comp': '[' in sol_code and 'for' in sol_code and ']' in sol_code,
                'uses_for_loop': 'for ' in sol_code,
                'uses_while_loop': 'while ' in sol_code,
                'uses_if': 'if ' in sol_code,
                'uses_elif': 'elif ' in sol_code,
                'uses_function_def': 'def ' in sol_code,
            }

            # 유사도 계산 (일치하는 패턴 개수)
            score = sum(1 for key in user_patterns if user_patterns[key] == sol_patterns[key])

            if score > max_score:
                max_score = score
                best_match = solution

        # 매칭 결과 없으면 첫 번째 solution 반환
        return best_match if best_match else solutions[0]

    def save_evaluation(self, comments: str, *ratings):
        """평가 저장"""
        if not self.current_hints:
            return "❌ 평가할 힌트가 없습니다. 먼저 힌트를 요청해주세요."

        available_models = self.model_manager.get_available_models()

        # 평가 데이터 구성
        evaluation = {
            'problem_id': self.current_problem['problem_id'],
            'problem_title': self.current_problem['title'],
            'timestamp': datetime.now().isoformat(),
            'ratings': {},
            'comments': comments
        }

        for i, model_name in enumerate(available_models):
            if i < len(ratings):
                evaluation['ratings'][model_name] = int(ratings[i])

        self.evaluation_results.append(evaluation)

        # 파일로 저장
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = os.path.join(self.results_dir, f"evaluation_{timestamp}.json")

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.evaluation_results, f, ensure_ascii=False, indent=2)

        return f"✅ 평가 저장 완료! (총 {len(self.evaluation_results)}개)\n📁 {filepath}"

    def _empty_state(self):
        """빈 상태 반환"""
        return "문제를 선택하고 '불러오기'를 클릭하세요.", gr.update(value=""), *self._empty_hints()

    def _empty_hints(self):
        """빈 힌트 UI 반환 (5개 모델 슬롯 = 10개 컴포넌트)"""
        hints = []
        ratings = []
        for _ in range(5):
            hints.append(gr.update(value="", visible=False))   # hint_text (Markdown)
            ratings.append(gr.update(value=3, visible=False))  # rating (Slider)
        return hints + ratings  # hints 먼저, ratings 나중


def create_single_page_ui(app: HintEvaluationApp):
    """단일 페이지 UI"""

    with gr.Blocks(title="코딩 힌트 평가 시스템", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🎯 코딩 힌트 모델 평가 시스템")
        gr.Markdown("문제 선택 → 코드 작성 → 힌트 받기 → 평가하기를 한 화면에서!")

        # 상태 메시지
        status_msg = gr.Markdown("👋 문제를 선택해주세요.")

        # ===== 섹션 1: 문제 선택 =====
        with gr.Row():
            problem_dropdown = gr.Dropdown(
                choices=app.get_problem_list(),
                label="📚 문제 선택",
                interactive=True,
                scale=3
            )
            load_btn = gr.Button("📂 문제 불러오기", variant="primary", scale=1)

        # ===== 섹션 2: 문제 표시 =====
        problem_display = gr.Markdown("", visible=True)

        gr.Markdown("---")

        # ===== 섹션 3: 코드 작성 =====
        gr.Markdown("## 💻 당신의 코드")
        user_code = gr.Code(
            label="Python 코드 작성",
            language="python",
            lines=12,
            value="# 여기에 코드를 작성하세요\n"
        )

        # 모델 선택 체크박스
        gr.Markdown("### 🎯 힌트를 받을 모델 선택")
        available_models = app.model_manager.get_available_models()

        model_checkboxes = []
        with gr.Row():
            for model_name in available_models[:5]:
                checkbox = gr.Checkbox(
                    label=model_name,
                    value=True,  # 기본 선택
                    interactive=True
                )
                model_checkboxes.append(checkbox)

        # Temperature 조절 슬라이더
        gr.Markdown("### 🌡️ Temperature 조절 (창의성 vs 일관성)")
        temperature_slider = gr.Slider(
            minimum=0.1,
            maximum=1.0,
            value=0.8,
            step=0.05,
            label="Temperature",
            info="낮을수록 일관적/결정론적, 높을수록 창의적/다양함",
            interactive=True
        )

        hint_btn = gr.Button("💡 선택한 모델로 힌트 요청하기", variant="primary", size="lg")

        gr.Markdown("---")

        # ===== 섹션 4: 모델별 힌트 & 평가 =====
        gr.Markdown("## 🤖 모델별 힌트 및 평가")
        gr.Markdown("각 모델의 힌트를 읽고 별점을 매겨주세요!")

        available_models = app.model_manager.get_available_models()

        # 동적으로 모델별 카드 생성 (최대 5개)
        hint_components = []
        rating_components = []

        for i, model_name in enumerate(available_models[:5]):
            with gr.Group():
                gr.Markdown(f"### 📌 {model_name}")

                hint_text = gr.Markdown("_힌트가 여기에 표시됩니다._", visible=False)

                rating = gr.Slider(
                    minimum=1,
                    maximum=5,
                    value=3,
                    step=1,
                    label="⭐ 평가 (1=최악, 5=최고)",
                    info="이 힌트가 얼마나 도움이 되었나요?",
                    visible=False
                )

                hint_components.append(hint_text)
                rating_components.append(rating)

        # 남은 슬롯 (숨김)
        for i in range(len(available_models), 5):
            hint_text = gr.Markdown("", visible=False)
            rating = gr.Slider(1, 5, 3, visible=False)
            hint_components.append(hint_text)
            rating_components.append(rating)

        gr.Markdown("---")

        # ===== 섹션 5: 코멘트 & 저장 =====
        gr.Markdown("## 📝 종합 평가")
        comments = gr.Textbox(
            label="전체 코멘트 (선택)",
            placeholder="각 모델의 힌트에 대한 의견을 자유롭게 작성하세요...",
            lines=3
        )

        save_btn = gr.Button("💾 평가 저장", variant="stop", size="lg")
        save_result = gr.Textbox(label="저장 결과", interactive=False)

        # ===== 이벤트 핸들러 =====

        # 문제 불러오기
        load_btn.click(
            fn=app.load_problem,
            inputs=[problem_dropdown],
            outputs=[problem_display, user_code] + hint_components + rating_components
        )

        # 힌트 요청 (temperature 포함)
        hint_btn.click(
            fn=app.request_hints,
            inputs=[user_code, temperature_slider] + model_checkboxes,
            outputs=[status_msg] + hint_components + rating_components
        )

        # 평가 저장
        save_btn.click(
            fn=app.save_evaluation,
            inputs=[comments] + rating_components,
            outputs=[save_result]
        )

    return demo


if __name__ == "__main__":
    # 명령행 인자 파싱
    parser = argparse.ArgumentParser(description="vLLM Hint Generation System")
    parser.add_argument("--server-name", type=str, default=None,
                       help="Server host (default: 127.0.0.1, use 0.0.0.0 for RunPod)")
    parser.add_argument("--server-port", type=int, default=7860,
                       help="Server port (default: 7860)")
    parser.add_argument("--share", action="store_true",
                       help="Create public share link")
    parser.add_argument("--no-browser", action="store_true",
                       help="Don't auto-open browser")
    args = parser.parse_args()

    print("\n" + "🎯" * 30)
    print("코딩 힌트 모델 평가 시스템 v5 - vLLM 고속 추론")
    print("🎯" * 30 + "\n")

    # RunPod 환경 자동 감지
    is_runpod = os.getenv("RUNPOD_POD_ID") is not None or os.getenv("PUBLIC_URL") is not None

    if is_runpod and args.server_name is None:
        args.server_name = "0.0.0.0"
        args.share = True
        args.no_browser = True
        print("🚀 RunPod 환경 감지됨!")
        print(f"   Public URL로 접속 가능합니다.\n")

    # 환경 설정 출력
    Config.print_config()

    # 데이터 경로 확인 (config에서 자동으로 가져옴)
    DATA_PATH = Config.DATA_FILE_PATH
    if not DATA_PATH.exists():
        print(f"❌ 데이터 파일을 찾을 수 없습니다: {DATA_PATH}")
        print(f"💡 .env 파일에서 DATA_FILE_PATH를 확인하거나")
        print(f"   {Config.get_relative_path(DATA_PATH)} 위치에 파일을 배치하세요.")
        exit(1)

    # 앱 초기화
    print("\n📚 문제 데이터 로딩 중...")
    app = HintEvaluationApp(str(DATA_PATH), auto_setup_models=True)
    print(f"✅ {len(app.problems)}개 문제 로드 완료!\n")

    # UI 생성 및 실행
    print("🌐 Gradio UI 시작 중...\n")
    demo = create_single_page_ui(app)

    # Launch 설정
    launch_kwargs = {
        "server_port": args.server_port,
        "share": args.share,
        "inbrowser": not args.no_browser
    }

    if args.server_name:
        launch_kwargs["server_name"] = args.server_name

    demo.launch(**launch_kwargs)
