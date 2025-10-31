"""
전체 백준 단계별 문제 하이브리드 크롤링 스크립트
- 백준: 단계, 문제 설명
- solved.ac: 태그, 난이도
"""

from baekjoon_hybrid_crawler import BaekjoonHybridCrawler
import time
import sys
from pathlib import Path

# 프로젝트 루트를 sys.path에 추가 (config.py import를 위해)
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from config import Config


def main():
    print("="*70)
    print("  백준 전체 단계 하이브리드 크롤링 (Step 1 ~ 68)")
    print("  - 백준: 단계, 문제 설명")
    print("  - solved.ac: 태그, 난이도")
    print("="*70)
    print()

    # Config 출력
    print(f"출력 디렉토리: {Config.get_relative_path(Config.CRAWLER_OUTPUT_DIR)}\n")

    # output_dir을 지정하지 않으면 config에서 자동으로 가져옴
    crawler = BaekjoonHybridCrawler()

    # 먼저 모든 단계 정보 가져오기
    print("단계 목록 가져오는 중...")
    steps = crawler.get_step_list()

    if not steps:
        print("[ERROR] 단계 목록을 가져올 수 없습니다.")
        return

    print(f"[OK] 총 {len(steps)}개 단계 발견\n")

    # 사용자 확인
    print(f"총 {len(steps)}개 단계의 모든 문제를 크롤링합니다.")
    print("예상 시간: 약 30-60분 (문제 개수에 따라 다름)")
    print("주의: 서버 부하 방지를 위해 각 요청 사이에 1초씩 대기합니다.\n")

    choice = input("계속하시겠습니까? (y/n): ").strip().lower()

    if choice != 'y':
        print("취소되었습니다.")
        return

    # 전체 크롤링 시작
    print("\n" + "="*70)
    print("하이브리드 크롤링 시작")
    print("="*70 + "\n")

    start_time = time.time()

    # Step 1부터 68까지 크롤링
    all_problems = crawler.crawl_by_steps(
        start_step=1,
        end_step=68,
        delay=1.0  # 1초 대기
    )

    end_time = time.time()
    elapsed = end_time - start_time

    # 최종 결과
    print("\n" + "="*70)
    print("크롤링 완료!")
    print("="*70)
    print(f"총 문제 수: {len(all_problems)}개")
    print(f"소요 시간: {elapsed/60:.1f}분 ({elapsed:.0f}초)")
    print(f"저장 위치: {Config.get_relative_path(Config.CRAWLER_OUTPUT_DIR)}/problems_hybrid_step_1_to_68.json")

    # 통계
    total_tags = sum(len(p['tags']) for p in all_problems)
    problems_with_tags = sum(1 for p in all_problems if p['tags'])

    print(f"\n통계:")
    print(f"  - 태그가 있는 문제: {problems_with_tags}개 ({problems_with_tags/len(all_problems)*100:.1f}%)")
    print(f"  - 총 태그 수: {total_tags}개")
    print(f"  - 평균 태그 수: {total_tags/len(all_problems):.2f}개/문제")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
