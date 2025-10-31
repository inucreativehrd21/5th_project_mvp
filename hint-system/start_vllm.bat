@echo off
REM vLLM Server Launch Script for Windows
REM
REM This script starts a vLLM server for fast hint generation.
REM Modify the MODEL variable below to change the model.

echo ============================================================
echo Starting vLLM Server for Code Hint Generation
echo ============================================================

REM Configuration
set MODEL=Qwen/Qwen2.5-Coder-7B-Instruct
set PORT=8000
set GPU_MEMORY_UTILIZATION=0.9
set MAX_MODEL_LEN=4096

echo Model: %MODEL%
echo Port: %PORT%
echo GPU Memory: %GPU_MEMORY_UTILIZATION%
echo Max Length: %MAX_MODEL_LEN%
echo ============================================================

REM Activate virtual environment if it exists
if exist venv\Scripts\activate.bat (
    echo Activating virtual environment...
    call venv\Scripts\activate.bat
)

REM Start vLLM server
python vllm_server.py ^
    --model %MODEL% ^
    --port %PORT% ^
    --gpu-memory-utilization %GPU_MEMORY_UTILIZATION% ^
    --max-model-len %MAX_MODEL_LEN% ^
    --trust-remote-code

pause
