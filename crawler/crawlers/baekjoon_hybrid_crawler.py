"""
백준 + solved.ac 하이브리드 크롤러
- 백준: 단계별 문제 목록, 문제 설명, 예제
- solved.ac: 문제 태그(분류), 난이도
"""

import requests
from bs4 import BeautifulSoup
import json
import time
import sys
from typing import Dict, List, Optional
from pathlib import Path

# 프로젝트 루트를 sys.path에 추가 (config.py import를 위해)
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from config import Config


class BaekjoonHybridCrawler:
    """백준 + solved.ac 하이브리드 크롤러"""

    BAEKJOON_URL = "https://www.acmicpc.net"
    SOLVED_AC_URL = "https://solved.ac/api/v3"

    def __init__(self, output_dir: str = None):
        # output_dir이 지정되지 않으면 config에서 자동으로 가져옴
        if output_dir is None:
            self.output_dir = Config.CRAWLER_OUTPUT_DIR
        else:
            self.output_dir = Path(output_dir)

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 백준용 세션
        self.baekjoon_session = requests.Session()
        self.baekjoon_session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })

        # solved.ac용 세션
        self.solved_session = requests.Session()

    def get_step_list(self) -> List[Dict]:
        """백준에서 단계 목록 가져오기"""
        try:
            print("단계 목록 가져오는 중...")
            url = f"{self.BAEKJOON_URL}/step"
            response = self.baekjoon_session.get(url, timeout=10)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, 'html.parser')
            steps = []

            table = soup.find('table', class_='table-bordered')
            if not table:
                print("[ERROR] 단계 목록 테이블을 찾을 수 없습니다.")
                return steps

            tbody = table.find('tbody')
            if not tbody:
                return steps

            rows = tbody.find_all('tr')

            for row in rows:
                cols = row.find_all('td')
                if len(cols) >= 3:
                    # 단계 순서 (테이블의 "단계" 컬럼 값)
                    step_order = cols[0].text.strip()

                    link = cols[1].find('a')
                    if link:
                        title = link.text.strip()
                        # href에서 실제 URL ID 추출 (예: /step/1 -> 1)
                        href = link.get('href', '')
                        step_url_id = href.split('/')[-1] if href else step_order

                        description = cols[2].text.strip() if len(cols) > 2 else ""

                        steps.append({
                            "step": int(step_order),  # JSON에 저장될 단계 번호
                            "step_url_id": int(step_url_id),  # URL 접근용 ID
                            "title": title,
                            "description": description
                        })

            print(f"[OK] {len(steps)}개 단계 발견")
            return steps

        except Exception as e:
            print(f"[ERROR] 오류: {e}")
            return []

    def get_problems_in_step(self, step_id: int) -> List[Dict]:
        """백준에서 특정 단계의 문제 목록 가져오기"""
        try:
            url = f"{self.BAEKJOON_URL}/step/{step_id}"
            print(f"\n[Step {step_id}] 문제 목록 가져오는 중...")

            response = self.baekjoon_session.get(url, timeout=10)

            if response.status_code == 404:
                print(f"  [SKIP] Step {step_id}는 존재하지 않습니다.")
                return []

            response.raise_for_status()

            soup = BeautifulSoup(response.text, 'html.parser')
            problems = []

            table = soup.find('table', id='problemset')
            if not table:
                print("  [ERROR] 문제 테이블을 찾을 수 없습니다.")
                return problems

            tbody = table.find('tbody')
            if not tbody:
                return problems

            rows = tbody.find_all('tr')

            for row in rows:
                cols = row.find_all('td')

                # 문제 정보 행
                if len(cols) >= 3 and not row.find('td', {'colspan': True}):
                    problem_id_td = row.find('td', class_='list_problem_id')
                    if problem_id_td:
                        problem_id = problem_id_td.text.strip()
                        title_link = row.find('a', href=lambda x: x and '/problem/' in x)
                        if title_link:
                            title = title_link.text.strip()
                            problems.append({
                                "problem_id": problem_id,
                                "title": title
                            })

            print(f"  [OK] {len(problems)}개 문제 발견")
            return problems

        except Exception as e:
            print(f"  [ERROR] 오류: {e}")
            return []

    def get_problem_description(self, problem_id: str) -> Optional[Dict]:
        """백준에서 문제 설명 가져오기"""
        try:
            url = f"{self.BAEKJOON_URL}/problem/{problem_id}"
            response = self.baekjoon_session.get(url, timeout=10)

            if response.status_code != 200:
                return None

            soup = BeautifulSoup(response.text, 'html.parser')

            # 문제 설명
            desc_elem = soup.find('div', {'id': 'problem_description'})
            description = desc_elem.get_text(strip=True) if desc_elem else ""

            # 입력 설명
            input_elem = soup.find('div', {'id': 'problem_input'})
            input_desc = input_elem.get_text(strip=True) if input_elem else ""

            # 출력 설명
            output_elem = soup.find('div', {'id': 'problem_output'})
            output_desc = output_elem.get_text(strip=True) if output_elem else ""

            # 예제
            examples = []
            sample_inputs = soup.find_all('pre', id=lambda x: x and 'sample-input' in x)
            sample_outputs = soup.find_all('pre', id=lambda x: x and 'sample-output' in x)

            for inp, out in zip(sample_inputs, sample_outputs):
                examples.append({
                    "input": inp.text.strip(),
                    "output": out.text.strip()
                })

            return {
                "description": description,
                "input_description": input_desc,
                "output_description": output_desc,
                "examples": examples
            }

        except Exception as e:
            return None

    def get_solved_ac_info(self, problem_ids: List[str]) -> Dict[str, Dict]:
        """solved.ac API에서 문제 정보 가져오기 (최대 100개씩)"""
        result = {}

        # solved.ac API는 한 번에 최대 100개까지
        chunk_size = 100

        for i in range(0, len(problem_ids), chunk_size):
            chunk = problem_ids[i:i + chunk_size]

            try:
                url = f"{self.SOLVED_AC_URL}/problem/lookup"
                params = {"problemIds": ",".join(chunk)}
                response = self.solved_session.get(url, params=params, timeout=10)

                if response.status_code == 200:
                    problems = response.json()

                    for prob in problems:
                        problem_id = str(prob['problemId'])
                        tags = [tag['displayNames'][0]['name'] for tag in prob.get('tags', [])]

                        result[problem_id] = {
                            "level": prob.get('level', 0),
                            "tags": tags,
                            "accepted_user_count": prob.get('acceptedUserCount', 0),
                            "average_tries": prob.get('averageTries', 0)
                        }
                else:
                    print(f"  [WARNING] solved.ac API 오류: {response.status_code}")

                time.sleep(0.5)  # API 요청 간격

            except Exception as e:
                print(f"  [WARNING] solved.ac API 오류: {e}")

        return result

    def crawl_by_steps(
        self,
        start_step: int = 1,
        end_step: int = 68,
        delay: float = 1.0
    ):
        """
        하이브리드 크롤링: 백준 + solved.ac
        """
        print(f"\n{'='*70}")
        print(f"하이브리드 크롤링 시작 (Step {start_step} ~ {end_step})")
        print(f"{'='*70}\n")

        # 1. 단계 목록 가져오기
        steps = self.get_step_list()
        if not steps:
            print("단계 목록을 가져올 수 없습니다.")
            return []

        all_problems = []
        total_success = 0
        total_failed = 0

        # 2. 각 단계별로 처리
        for step in steps:
            step_order = step['step']  # 단계 번호 (1, 2, 3, 4, 5, 6...)
            step_url_id = step['step_url_id']  # URL ID

            if step_order < start_step or step_order > end_step:
                continue

            print(f"\n{'─'*70}")
            print(f"Step {step_order}: {step['title']}")
            print(f"{'─'*70}")

            # 2-1. 백준에서 문제 목록 가져오기 (URL ID 사용)
            problems = self.get_problems_in_step(step_url_id)

            if not problems:
                continue

            # 2-2. solved.ac에서 태그 정보 가져오기
            print(f"  [solved.ac] {len(problems)}개 문제의 태그 정보 가져오는 중...")
            problem_ids = [p['problem_id'] for p in problems]
            solved_info = self.get_solved_ac_info(problem_ids)
            print(f"  [OK] {len(solved_info)}개 문제의 태그 정보 획득")

            # 2-3. 각 문제의 상세 정보 가져오기
            for idx, problem in enumerate(problems, 1):
                problem_id = problem['problem_id']

                print(f"  [{idx}/{len(problems)}] #{problem_id} {problem['title']}...", end=" ")

                # 백준에서 문제 설명 가져오기
                detail = self.get_problem_description(problem_id)

                if detail:
                    # solved.ac 정보 추가
                    solved = solved_info.get(problem_id, {})

                    # 최종 데이터 구성
                    combined = {
                        "problem_id": problem_id,
                        "title": problem['title'],
                        "level": solved.get('level', 0),
                        "tags": solved.get('tags', []),
                        "description": detail['description'],
                        "input_description": detail['input_description'],
                        "output_description": detail['output_description'],
                        "examples": detail['examples'],
                        "step": step_order,  # 단계 번호 (테이블의 "단계" 컬럼 값)
                        "step_title": step['title'],
                        "accepted_user_count": solved.get('accepted_user_count', 0),
                        "average_tries": solved.get('average_tries', 0),
                        "url": f"{self.BAEKJOON_URL}/problem/{problem_id}"
                    }

                    all_problems.append(combined)
                    total_success += 1
                    print("[OK]")
                else:
                    total_failed += 1
                    print("[FAIL]")

                time.sleep(delay)

        # 3. 저장
        if all_problems:
            filename = f"problems_hybrid_step_{start_step}_to_{end_step}.json"
            self._save_problems(all_problems, filename)

        print(f"\n{'='*70}")
        print(f"결과:")
        print(f"  [SUCCESS] 성공: {total_success}")
        print(f"  [FAIL] 실패: {total_failed}")
        print(f"  총 {len(all_problems)}개 문제 저장")
        print(f"{'='*70}\n")

        return all_problems

    def _save_problems(self, problems: List[Dict], filename: str):
        """JSON 파일로 저장"""
        filepath = self.output_dir / filename
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(problems, f, ensure_ascii=False, indent=2)

        print(f"\n→ 저장 위치: {filepath}")


def main():
    """메인 함수"""
    crawler = BaekjoonHybridCrawler(output_dir="../data/raw")

    print("="*70)
    print("  백준 + solved.ac 하이브리드 크롤러")
    print("  - 백준: 단계, 문제 설명")
    print("  - solved.ac: 태그, 난이도")
    print("="*70)
    print()

    print("크롤링 옵션:")
    print("1. Step 1~3 (빠른 테스트)")
    print("2. Step 1~10 (추천)")
    print("3. Step 1~68 (전체)")
    print("4. 직접 입력")
    print()

    choice = input("선택 (1-4): ").strip()

    if choice == "1":
        crawler.crawl_by_steps(1, 3, delay=1.0)
    elif choice == "2":
        crawler.crawl_by_steps(1, 10, delay=1.0)
    elif choice == "3":
        crawler.crawl_by_steps(1, 68, delay=1.0)
    elif choice == "4":
        start = int(input("시작 단계: "))
        end = int(input("종료 단계: "))
        crawler.crawl_by_steps(start, end, delay=1.0)
    else:
        print("잘못된 선택입니다.")


if __name__ == "__main__":
    main()
