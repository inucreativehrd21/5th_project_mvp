"""
프로젝트 환경 설정 관리
.env 파일에서 환경변수를 읽어와서 경로를 자동으로 설정합니다.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()


class Config:
    """프로젝트 설정 클래스"""

    # ========================================
    # 1. 프로젝트 루트 경로
    # ========================================
    # .env에서 읽거나, 없으면 현재 파일 위치 기준으로 자동 설정
    PROJECT_ROOT = Path(os.getenv(
        'PROJECT_ROOT',
        Path(__file__).parent.absolute()
    ))

    # ========================================
    # 2. 데이터 경로
    # ========================================
    # 힌트 시스템 데이터 파일
    DATA_FILE_PATH = Path(os.getenv(
        'DATA_FILE_PATH',
        PROJECT_ROOT / 'hint-system' / 'data' / 'problems_multi_solution.json'
    ))

    # 크롤러 출력 디렉토리
    CRAWLER_OUTPUT_DIR = Path(os.getenv(
        'CRAWLER_OUTPUT_DIR',
        PROJECT_ROOT / 'crawler' / 'data' / 'raw'
    ))

    # ========================================
    # 3. 평가 결과 저장 경로
    # ========================================
    EVALUATION_RESULTS_DIR = Path(os.getenv(
        'EVALUATION_RESULTS_DIR',
        PROJECT_ROOT / 'hint-system' / 'evaluation' / 'results'
    ))

    # ========================================
    # 4. 모델 설정
    # ========================================
    VLLM_SERVER_URL = os.getenv('VLLM_SERVER_URL', 'http://localhost:8000')
    DEFAULT_MODEL = os.getenv('DEFAULT_MODEL', 'Qwen/Qwen2.5-Coder-1.5B-Instruct')
    DEFAULT_TEMPERATURE = float(os.getenv('DEFAULT_TEMPERATURE', '0.7'))

    # ========================================
    # 5. API 키
    # ========================================
    SOLVED_AC_API_KEY = os.getenv('SOLVED_AC_API_KEY', '')

    # ========================================
    # 6. 로그 설정
    # ========================================
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    LOG_FILE_PATH = Path(os.getenv(
        'LOG_FILE_PATH',
        PROJECT_ROOT / 'logs' / 'app.log'
    ))

    @classmethod
    def create_directories(cls):
        """필요한 디렉토리 자동 생성"""
        directories = [
            cls.CRAWLER_OUTPUT_DIR,
            cls.EVALUATION_RESULTS_DIR,
            cls.LOG_FILE_PATH.parent,
            cls.DATA_FILE_PATH.parent
        ]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

    @classmethod
    def get_relative_path(cls, path: Path) -> str:
        """절대 경로를 프로젝트 루트 기준 상대 경로로 변환"""
        try:
            return str(path.relative_to(cls.PROJECT_ROOT))
        except ValueError:
            return str(path)

    @classmethod
    def print_config(cls):
        """현재 설정 출력 (디버깅용)"""
        print("=" * 60)
        print("프로젝트 환경 설정")
        print("=" * 60)
        print(f"프로젝트 루트: {cls.PROJECT_ROOT}")
        print(f"데이터 파일: {cls.get_relative_path(cls.DATA_FILE_PATH)}")
        print(f"크롤러 출력: {cls.get_relative_path(cls.CRAWLER_OUTPUT_DIR)}")
        print(f"평가 결과: {cls.get_relative_path(cls.EVALUATION_RESULTS_DIR)}")
        print(f"로그 파일: {cls.get_relative_path(cls.LOG_FILE_PATH)}")
        print("=" * 60)


# 앱 시작 시 필요한 디렉토리 자동 생성
Config.create_directories()


# ========================================
# 편의 함수
# ========================================

def get_data_path() -> Path:
    """데이터 파일 경로 반환"""
    return Config.DATA_FILE_PATH


def get_crawler_output_dir() -> Path:
    """크롤러 출력 디렉토리 반환"""
    return Config.CRAWLER_OUTPUT_DIR


def get_results_dir() -> Path:
    """평가 결과 디렉토리 반환"""
    return Config.EVALUATION_RESULTS_DIR


if __name__ == '__main__':
    # 설정 확인용
    Config.print_config()
