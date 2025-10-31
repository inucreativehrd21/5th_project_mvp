# RunPod 1분 빠른 시작

## 📋 필요한 것
- RunPod 계정
- GPU Pod (RTX 3090/4090 또는 A40 권장)
- 약 $0.30-0.40/시간 비용

---

## 🚀 3단계로 시작하기

### 1️⃣ RunPod Pod 생성
```
1. runpod.io 로그인
2. Deploy 클릭
3. GPU 선택: RTX 4090 (24GB) 권장
4. Template: PyTorch 또는 CUDA 12.1
5. Deploy!
```

### 2️⃣ 터미널에서 설치
```bash
cd /workspace
git clone <your-repo> hint-system
cd hint-system/hint-system
bash setup_runpod.sh
```

### 3️⃣ 서버 실행
```bash
# 터미널 1
source venv/bin/activate
./start_vllm.sh

# 터미널 2 (새 터미널)
source venv/bin/activate
./run_app.sh
```

---

## 🌐 접속
Gradio가 시작되면 표시되는 Public URL로 접속:
```
Running on public URL: https://xxxxx.gradio.live
```

---

## ⚡ 성능
- **힌트 생성**: 0.4-0.5초 (기존 대비 30배 빠름)
- **메모리**: 12GB (기존 18GB 대비 33% 절감)
- **품질**: 동기 유발 & 상상력 자극 프롬프트

---

## 🎯 GPU별 추천 모델

자동으로 감지되지만 수동 변경 가능:

| VRAM | 모델 | 비용/시간 |
|------|------|----------|
| 48GB+ | Qwen 32B | $1.69 (A100) |
| 24GB | Qwen 14B | $0.39 (4090) |
| 16GB | CodeLlama 13B | $0.34 (A40) |
| 8GB | Qwen 7B | $0.29 (3090) |

---

## 🐛 문제 해결

**Out of Memory?**
```bash
# 더 작은 모델 사용
VLLM_MODEL=Qwen/Qwen2.5-Coder-7B-Instruct ./start_vllm.sh
```

**연결 안됨?**
```bash
# 서버 상태 확인
curl http://localhost:8000/v1/models
```

**느림?**
```bash
# GPU 확인
nvidia-smi
```

---

## 💡 비용 절감 팁

1. **Spot Instance 사용**: 50-70% 저렴
2. **Idle Timeout 설정**: 1-2시간
3. **볼륨 재사용**: 모델 재다운로드 방지

---

## 📖 상세 가이드

- [전체 RunPod 가이드](hint-system/README_RUNPOD.md)
- [vLLM 가이드](hint-system/VLLM_GUIDE.md)
- [프롬프트 V6 설명](hint-system/README_VLLM.md)

---

시작할 준비 완료! 🎉
