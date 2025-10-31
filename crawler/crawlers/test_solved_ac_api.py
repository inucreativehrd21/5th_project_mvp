"""
solved.ac API 테스트
- 문제 정보 가져오기
- 태그 정보 포함
"""

import requests
import json

# 1. 단일 문제 정보 가져오기
print("="*70)
print("1. 단일 문제 정보 테스트 (Hello World - 2557)")
print("="*70)

problem_id = "2557"
url = f"https://solved.ac/api/v3/problem/show?problemId={problem_id}"

response = requests.get(url)
print(f"Status Code: {response.status_code}\n")

if response.status_code == 200:
    data = response.json()
    print(f"문제 번호: {data['problemId']}")
    print(f"제목(한글): {data['titleKo']}")
    print(f"난이도: {data.get('level', 'N/A')}")
    print(f"태그 개수: {len(data['tags'])}")
    print("태그:")
    for tag in data['tags']:
        korean_name = tag['displayNames'][0]['name']  # 한글 태그명
        print(f"  - {korean_name}")

    print(f"\n사용 가능한 모든 필드:")
    print(json.dumps(list(data.keys()), indent=2, ensure_ascii=False))
else:
    print(f"Error: {response.status_code}")

# 2. 여러 문제 정보 한 번에 가져오기
print("\n" + "="*70)
print("2. 여러 문제 정보 한 번에 가져오기")
print("="*70)

problem_ids = ["2557", "1000", "1001"]
url = "https://solved.ac/api/v3/problem/lookup"
params = {"problemIds": ",".join(problem_ids)}

response = requests.get(url, params=params)
print(f"Status Code: {response.status_code}\n")

if response.status_code == 200:
    problems = response.json()
    print(f"총 {len(problems)}개 문제 정보 가져옴:\n")
    for prob in problems:
        print(f"  #{prob['problemId']}: {prob['titleKo']} (태그: {len(prob['tags'])}개)")
else:
    print(f"Error: {response.status_code}")

# 3. 단계별 문제 목록은? (이건 백준에서 가져와야 함)
print("\n" + "="*70)
print("3. solved.ac에서 단계 정보 제공 여부")
print("="*70)
print("[결과] solved.ac는 '단계별' 분류를 제공하지 않습니다.")
print("단계(step) 정보는 백준에서만 제공됩니다.")
print("\n[결론]")
print("- 백준: 단계별 문제 목록 크롤링")
print("- solved.ac: 각 문제의 상세 정보 + 태그")
print("- 두 가지를 결합!")
