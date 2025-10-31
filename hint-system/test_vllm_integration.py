#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
vLLM 통합 테스트
실제 vLLM 서버와의 연동 테스트
"""

import sys
import os
from pathlib import Path

# UTF-8 출력 강제 (Windows 호환성)
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# 프로젝트 루트를 sys.path에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_vllm_server_connection():
    """vLLM 서버 연결 테스트"""
    print("=" * 60)
    print("vLLM 서버 연결 테스트")
    print("=" * 60)

    try:
        from config import Config
        from models.model_inference import VLLMInference

        vllm_url = Config.VLLM_SERVER_URL or "http://localhost:8000/v1"
        print(f"\nvLLM 서버 URL: {vllm_url}")

        # 연결 시도
        print("\n연결 시도 중...")
        vllm_client = VLLMInference(
            model_name="test",
            base_url=vllm_url
        )

        print("✓ vLLM 서버 연결 성공!")
        return True

    except Exception as e:
        print(f"✗ vLLM 서버 연결 실패: {e}")
        print("\nvLLM 서버가 실행 중인지 확인하세요:")
        print("  ./start_vllm.sh")
        return False

def test_hint_generation():
    """힌트 생성 테스트"""
    print("\n" + "=" * 60)
    print("힌트 생성 테스트")
    print("=" * 60)

    try:
        from config import Config
        from models.model_inference import VLLMInference

        vllm_url = Config.VLLM_SERVER_URL or "http://localhost:8000/v1"

        # VLLMInference 초기화
        vllm_client = VLLMInference(
            model_name="auto",
            base_url=vllm_url
        )

        # 테스트 프롬프트
        test_prompt = """
학생 코드:
A = int(input())
B = int(input())

다음 목표: 두 수의 합을 출력하기

학생이 스스로 깨닫도록 질문 1개를 만드세요.
"""

        print("\n프롬프트:")
        print(test_prompt)
        print("\n힌트 생성 중...")

        # 힌트 생성
        result = vllm_client.generate_hint(
            prompt=test_prompt,
            max_tokens=200,
            temperature=0.7
        )

        if result['error']:
            print(f"✗ 오류: {result['error']}")
            return False

        print(f"\n✓ 힌트 생성 성공!")
        print(f"  생성 시간: {result['time']:.2f}초")
        print(f"  토큰 수: {result['tokens']}")
        print(f"  완료 이유: {result['finish_reason']}")
        print(f"\n생성된 힌트:")
        print(f"  {result['hint']}")

        return True

    except Exception as e:
        print(f"✗ 힌트 생성 실패: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_model_manager():
    """ModelManager 통합 테스트"""
    print("\n" + "=" * 60)
    print("ModelManager 통합 테스트")
    print("=" * 60)

    try:
        from config import Config
        from models.model_inference import ModelManager

        manager = ModelManager(sequential_load=True)

        # vLLM 모델 추가
        vllm_url = Config.VLLM_SERVER_URL
        if vllm_url:
            print(f"\nvLLM 모델 추가 중... (URL: {vllm_url})")
            manager.add_vllm_model(
                model_name="vLLM-Server",
                model_path="auto",
                base_url=vllm_url
            )

            available = manager.get_available_models()
            print(f"✓ 사용 가능한 모델: {', '.join(available)}")

            # 테스트 힌트 생성
            print("\n테스트 힌트 생성 중...")
            results = manager.generate_hints(
                prompt="파이썬에서 반복문을 사용하려면?",
                selected_models=available[:1],  # 첫 번째 모델만
                max_tokens=100,
                temperature=0.7
            )

            for model_name, result in results.items():
                print(f"\n모델: {model_name}")
                print(f"  시간: {result['time']:.2f}초")
                if result['error']:
                    print(f"  오류: {result['error']}")
                else:
                    print(f"  힌트: {result['hint'][:100]}...")

            return True
        else:
            print("✗ VLLM_SERVER_URL이 설정되지 않았습니다.")
            return False

    except Exception as e:
        print(f"✗ ModelManager 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """전체 통합 테스트"""
    print("""
============================================================
   vLLM 통합 테스트
============================================================

이 테스트는 vLLM 서버가 실행 중이어야 합니다.

실행 방법:
1. 터미널 1: ./start_vllm.sh
2. 터미널 2: python test_vllm_integration.py
""")

    input("vLLM 서버가 실행 중이면 Enter를 누르세요... ")

    results = []

    # 서버 연결
    results.append(test_vllm_server_connection())

    if results[0]:
        # 힌트 생성
        results.append(test_hint_generation())

        # ModelManager
        results.append(test_model_manager())

    # 결과
    print("\n" + "=" * 60)
    print("테스트 결과 요약")
    print("=" * 60)

    passed = sum(results)
    total = len(results)

    if passed == total:
        print(f"\n✓ 모든 통합 테스트 통과! ({passed}/{total})")
        print("\nvLLM 통합이 정상 작동합니다!")
        return 0
    else:
        print(f"\n✗ 테스트 실패: {total - passed}/{total}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
