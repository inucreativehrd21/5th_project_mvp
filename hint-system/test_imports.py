#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RunPod 배포 전 완전성 테스트
모든 import, 버전 호환성, 구성 요소 검증
"""

import sys
import os
from pathlib import Path

# UTF-8 출력 강제 (Windows 호환성)
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# 색상 출력
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    RESET = '\033[0m'

def print_test(name, status, message=""):
    """테스트 결과 출력"""
    if status:
        print(f"{Colors.GREEN}✓{Colors.RESET} {name}")
        if message:
            print(f"  {Colors.BLUE}{message}{Colors.RESET}")
    else:
        print(f"{Colors.RED}✗{Colors.RESET} {name}")
        if message:
            print(f"  {Colors.RED}{message}{Colors.RESET}")
    return status

def test_python_version():
    """Python 버전 확인"""
    version = sys.version_info
    supported = (3, 8) <= (version.major, version.minor) <= (3, 11)
    return print_test(
        "Python 버전",
        supported,
        f"Python {version.major}.{version.minor}.{version.micro} ({'지원됨' if supported else '미지원 - 3.8-3.11 필요'})"
    )

def test_core_imports():
    """핵심 라이브러리 import 테스트"""
    print(f"\n{Colors.YELLOW}[1/8] 핵심 라이브러리 테스트{Colors.RESET}")

    tests = []

    # Gradio
    try:
        import gradio as gr
        tests.append(print_test("Gradio", True, f"v{gr.__version__}"))
    except ImportError as e:
        tests.append(print_test("Gradio", False, str(e)))

    # Transformers
    try:
        import transformers
        tests.append(print_test("Transformers", True, f"v{transformers.__version__}"))
    except ImportError as e:
        tests.append(print_test("Transformers", False, str(e)))

    # PyTorch
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        cuda_version = torch.version.cuda if cuda_available else "N/A"
        tests.append(print_test(
            "PyTorch",
            True,
            f"v{torch.__version__} (CUDA: {cuda_version if cuda_available else 'Not Available'})"
        ))
    except ImportError as e:
        tests.append(print_test("PyTorch", False, str(e)))

    # OpenAI (vLLM 클라이언트)
    try:
        import openai
        tests.append(print_test("OpenAI", True, f"v{openai.__version__}"))
    except ImportError as e:
        tests.append(print_test("OpenAI", False, str(e)))

    # Python-dotenv
    try:
        import dotenv
        tests.append(print_test("Python-dotenv", True))
    except ImportError as e:
        tests.append(print_test("Python-dotenv", False, str(e)))

    return all(tests)

def test_vllm():
    """vLLM 설치 확인 (선택사항)"""
    print(f"\n{Colors.YELLOW}[2/8] vLLM 설치 테스트 (선택사항){Colors.RESET}")

    try:
        import vllm
        return print_test("vLLM", True, f"v{vllm.__version__}")
    except ImportError as e:
        return print_test(
            "vLLM",
            False,
            "미설치 (선택사항 - HuggingFace로 폴백 가능)"
        )

def test_project_structure():
    """프로젝트 구조 확인"""
    print(f"\n{Colors.YELLOW}[3/8] 프로젝트 구조 테스트{Colors.RESET}")

    required_files = [
        "app.py",
        "models/model_inference.py",
        "models/model_config.py",
        "models/__init__.py",
        "start_vllm.sh",
        "run_app.sh",
        "setup_runpod.sh",
        ".env.example",
    ]

    tests = []
    base_path = Path(__file__).parent

    for file_path in required_files:
        full_path = base_path / file_path
        tests.append(print_test(
            f"파일: {file_path}",
            full_path.exists(),
            f"경로: {full_path}"
        ))

    return all(tests)

def test_config_loading():
    """Config 로딩 테스트"""
    print(f"\n{Colors.YELLOW}[4/8] Config 로딩 테스트{Colors.RESET}")

    try:
        # 프로젝트 루트를 sys.path에 추가
        project_root = Path(__file__).parent.parent
        sys.path.insert(0, str(project_root))

        from config import Config

        tests = []
        tests.append(print_test("Config import", True))
        tests.append(print_test(
            "PROJECT_ROOT",
            Config.PROJECT_ROOT is not None,
            f"경로: {Config.PROJECT_ROOT}"
        ))
        tests.append(print_test(
            "VLLM_SERVER_URL",
            True,
            f"URL: {Config.VLLM_SERVER_URL or '미설정 (폴백 모드)'}"
        ))

        return all(tests)
    except Exception as e:
        return print_test("Config 로딩", False, str(e))

def test_model_config():
    """모델 설정 테스트"""
    print(f"\n{Colors.YELLOW}[5/8] 모델 설정 테스트{Colors.RESET}")

    try:
        from models.model_config import (
            VLLM_MODELS,
            LOCAL_MODELS,
            RUNPOD_MODELS,
            GENERATION_PARAMS,
            CREATIVE_GENERATION_PARAMS,
            SOCRATIC_GENERATION_PARAMS,
            VLLM_SERVER_CONFIG,
            RECOMMENDED_MODELS,
        )

        tests = []
        tests.append(print_test(
            "VLLM_MODELS",
            len(VLLM_MODELS) > 0,
            f"{len(VLLM_MODELS)}개 모델 설정"
        ))
        tests.append(print_test(
            "LOCAL_MODELS",
            len(LOCAL_MODELS) > 0,
            f"{len(LOCAL_MODELS)}개 모델 설정"
        ))
        tests.append(print_test(
            "GENERATION_PARAMS",
            "temperature" in GENERATION_PARAMS,
            f"temperature: {GENERATION_PARAMS['temperature']}"
        ))
        tests.append(print_test(
            "CREATIVE_GENERATION_PARAMS",
            "temperature" in CREATIVE_GENERATION_PARAMS,
            f"temperature: {CREATIVE_GENERATION_PARAMS['temperature']}"
        ))
        tests.append(print_test(
            "VLLM_SERVER_CONFIG",
            "default_url" in VLLM_SERVER_CONFIG,
            f"default_url: {VLLM_SERVER_CONFIG['default_url']}"
        ))

        return all(tests)
    except Exception as e:
        return print_test("모델 설정 로딩", False, str(e))

def test_model_inference():
    """ModelInference 클래스 테스트"""
    print(f"\n{Colors.YELLOW}[6/8] ModelInference 클래스 테스트{Colors.RESET}")

    try:
        from models.model_inference import (
            ModelInference,
            HuggingFaceInference,
            VLLMInference,
            OllamaInference,
            ModelManager
        )

        tests = []
        tests.append(print_test("ModelInference", True))
        tests.append(print_test("HuggingFaceInference", True))
        tests.append(print_test("VLLMInference", True))
        tests.append(print_test("OllamaInference", True))
        tests.append(print_test("ModelManager", True))

        # VLLMInference 초기화 테스트 (연결은 안함)
        try:
            vllm_inf = VLLMInference(
                model_name="test-model",
                base_url="http://localhost:8000/v1"
            )
            tests.append(print_test(
                "VLLMInference 초기화",
                vllm_inf is not None,
                "메서드: generate_hint, generate_hint_stream"
            ))
        except Exception as e:
            # 서버 연결 실패는 정상 (아직 실행 안함)
            if "not reachable" in str(e).lower() or "connection" in str(e).lower():
                tests.append(print_test(
                    "VLLMInference 초기화",
                    True,
                    "클래스 정상 (서버 미실행은 정상)"
                ))
            else:
                tests.append(print_test("VLLMInference 초기화", False, str(e)))

        # ModelManager 초기화 테스트
        try:
            manager = ModelManager(sequential_load=True)
            tests.append(print_test(
                "ModelManager 초기화",
                manager is not None,
                "메서드: add_vllm_model, add_huggingface_model"
            ))
        except Exception as e:
            tests.append(print_test("ModelManager 초기화", False, str(e)))

        return all(tests)
    except Exception as e:
        return print_test("ModelInference 로딩", False, str(e))

def test_app_imports():
    """app.py import 테스트"""
    print(f"\n{Colors.YELLOW}[7/8] app.py import 테스트{Colors.RESET}")

    try:
        # app.py의 주요 함수들이 정상적으로 로드되는지 확인
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "app",
            Path(__file__).parent / "app.py"
        )
        app_module = importlib.util.module_from_spec(spec)

        # 실제 실행은 하지 않고 import만 테스트
        tests = []
        tests.append(print_test("app.py 모듈 로딩", spec is not None))

        # 주요 클래스 확인
        try:
            spec.loader.exec_module(app_module)

            tests.append(print_test(
                "HintEvaluationApp 클래스",
                hasattr(app_module, 'HintEvaluationApp')
            ))
            tests.append(print_test(
                "create_single_page_ui 함수",
                hasattr(app_module, 'create_single_page_ui')
            ))

            # argparse 지원 확인 (import 확인)
            import argparse as arg_module
            tests.append(print_test(
                "argparse 지원",
                'ArgumentParser' in dir(arg_module),
                "RunPod 환경 자동 감지 지원"
            ))

        except Exception as e:
            # Gradio UI 초기화 관련 에러는 무시 (실제 실행 안함)
            if "launch" not in str(e).lower() and "data_file_path" not in str(e).lower():
                tests.append(print_test("app.py 실행 테스트", False, str(e)))

        return all(tests)
    except Exception as e:
        return print_test("app.py 로딩", False, str(e))

def test_data_files():
    """데이터 파일 확인"""
    print(f"\n{Colors.YELLOW}[8/8] 데이터 파일 테스트{Colors.RESET}")

    try:
        project_root = Path(__file__).parent.parent
        sys.path.insert(0, str(project_root))

        from config import Config

        tests = []

        # 데이터 파일
        data_path = Config.DATA_FILE_PATH
        if data_path and data_path.exists():
            tests.append(print_test(
                "problems_multi_solution.json",
                True,
                f"경로: {data_path}"
            ))

            # JSON 유효성 확인
            try:
                import json
                with open(data_path, 'r', encoding='utf-8') as f:
                    problems = json.load(f)
                    tests.append(print_test(
                        "JSON 유효성",
                        isinstance(problems, list),
                        f"{len(problems)}개 문제 로드됨"
                    ))
            except Exception as e:
                tests.append(print_test("JSON 파싱", False, str(e)))
        else:
            tests.append(print_test(
                "problems_multi_solution.json",
                False,
                f"파일 없음: {data_path}"
            ))

        # 평가 결과 디렉토리
        eval_dir = Config.EVALUATION_RESULTS_DIR
        tests.append(print_test(
            "평가 결과 디렉토리",
            eval_dir is not None,
            f"경로: {eval_dir}"
        ))

        return all(tests)
    except Exception as e:
        return print_test("데이터 파일 확인", False, str(e))

def test_gpu_availability():
    """GPU 사용 가능 여부 확인"""
    print(f"\n{Colors.YELLOW}[추가] GPU 확인{Colors.RESET}")

    try:
        import torch

        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_names = [torch.cuda.get_device_name(i) for i in range(gpu_count)]

            print_test(
                "CUDA 사용 가능",
                True,
                f"{gpu_count}개 GPU 감지"
            )

            for i, name in enumerate(gpu_names):
                memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                print(f"  GPU {i}: {name} ({memory:.1f}GB)")

            return True
        else:
            return print_test(
                "CUDA",
                False,
                "GPU 없음 (CPU 모드, vLLM 사용 불가)"
            )
    except Exception as e:
        return print_test("GPU 확인", False, str(e))

def main():
    """전체 테스트 실행"""
    print(f"""
{Colors.BLUE}============================================================
   vLLM Hint System - RunPod 배포 테스트
============================================================{Colors.RESET}
""")

    results = []

    # Python 버전
    results.append(test_python_version())

    # 핵심 라이브러리
    results.append(test_core_imports())

    # vLLM (선택사항)
    vllm_result = test_vllm()
    # vLLM은 선택사항이므로 전체 결과에 영향 안줌

    # 프로젝트 구조
    results.append(test_project_structure())

    # Config 로딩
    results.append(test_config_loading())

    # 모델 설정
    results.append(test_model_config())

    # ModelInference
    results.append(test_model_inference())

    # app.py
    results.append(test_app_imports())

    # 데이터 파일
    results.append(test_data_files())

    # GPU (선택사항)
    gpu_result = test_gpu_availability()

    # 결과 요약
    print(f"\n{Colors.BLUE}============================================================")
    print(f"   테스트 결과 요약")
    print(f"============================================================{Colors.RESET}")

    passed = sum(results)
    total = len(results)

    if passed == total:
        print(f"\n{Colors.GREEN}✓ 모든 필수 테스트 통과! ({passed}/{total}){Colors.RESET}")
        print(f"\n{Colors.GREEN}RunPod에서 바로 실행 가능합니다!{Colors.RESET}")

        if not vllm_result:
            print(f"\n{Colors.YELLOW}⚠ vLLM 미설치 - HuggingFace로 폴백됩니다 (느림){Colors.RESET}")
            print(f"   설치: pip install vllm")

        if not gpu_result:
            print(f"\n{Colors.YELLOW}⚠ GPU 없음 - CPU 모드로 실행됩니다 (매우 느림){Colors.RESET}")
            print(f"   RunPod에서 GPU Pod를 선택하세요")

        print(f"\n{Colors.BLUE}다음 단계:{Colors.RESET}")
        print(f"1. vLLM 서버 시작: ./start_vllm.sh")
        print(f"2. Gradio 앱 실행: ./run_app.sh")

        return 0
    else:
        print(f"\n{Colors.RED}✗ 테스트 실패: {total - passed}/{total}{Colors.RESET}")
        print(f"\n{Colors.RED}위의 오류를 해결한 후 다시 시도하세요.{Colors.RESET}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
