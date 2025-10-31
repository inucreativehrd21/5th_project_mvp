@echo off
chcp 65001 >nul
echo ======================================================================
echo   백준 문제 크롤링 실행
echo ======================================================================
echo.

cd /d "%~dp0"

echo [선택 1] Step 1-3 크롤링 (빠른 테스트, ~1분)
echo [선택 2] Step 1-10 크롤링 (추천, ~5분)
echo [선택 3] Step 1-68 전체 크롤링 (~60분)
echo.

set /p choice="선택 (1/2/3): "

if "%choice%"=="1" (
    echo.
    echo Step 1-3 크롤링 시작...
    echo 1 | ..\..\..\..\..\.venv\Scripts\python.exe baekjoon_hybrid_crawler.py
) else if "%choice%"=="2" (
    echo.
    echo Step 1-10 크롤링 시작...
    echo 2 | ..\..\..\..\..\.venv\Scripts\python.exe baekjoon_hybrid_crawler.py
) else if "%choice%"=="3" (
    echo.
    echo Step 1-68 전체 크롤링 시작...
    ..\..\..\..\..\.venv\Scripts\python.exe crawl_all_hybrid.py
) else (
    echo 잘못된 선택입니다.
    pause
    exit /b
)

echo.
echo ======================================================================
echo 크롤링 완료!
echo 결과 파일: ..\data\raw\problems_hybrid_step_X_to_Y.json
echo ======================================================================
echo.
pause
