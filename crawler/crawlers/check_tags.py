import requests
from bs4 import BeautifulSoup

session = requests.Session()
session.headers.update({
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
})

# Hello World 문제 확인
url = 'https://www.acmicpc.net/problem/2557'
response = session.get(url)

soup = BeautifulSoup(response.text, 'html.parser')

# 태그 링크 찾기
tag_links = soup.find_all('a', href=lambda x: x and '/problem/tag/' in x)

print(f"URL: {url}")
print(f"태그 링크 개수: {len(tag_links)}")

if tag_links:
    print("발견된 태그:")
    for link in tag_links:
        print(f"  - {link.text.strip()}")
else:
    print("\n[결과] 태그를 찾을 수 없습니다.")
    print("이유: 백준은 비로그인 사용자에게 문제 분류(태그)를 보여주지 않습니다.")
    print("\n해결 방법:")
    print("1. 로그인 쿠키를 사용하여 크롤링")
    print("2. solved.ac API 사용 (태그 정보 포함)")
