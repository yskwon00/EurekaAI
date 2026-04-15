# 🧠 EurekaAI — Self-Learning Curriculum AI

> **"유레카!"** — 신생아처럼 백지 상태에서 시작해 스스로 초·중·고·대학·사회 수준으로 성장하는 AI

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-EE4C2C.svg)](https://pytorch.org)
[![Mac MPS](https://img.shields.io/badge/Mac-Apple%20Silicon-black.svg)](https://developer.apple.com/metal/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## ✨ 개요

EurekaAI는 **사전 학습된 지식 없이** 커리큘럼 기반으로 스스로 성장하는 한국어+영어 이중언어 AI 모델입니다.

```
Stage 0 → Stage 1 → Stage 2 → Stage 3 → Stage 4 → Stage 5 → Stage 6
🍼 신생아   🧸 유아   📚 초등   🔢 중등   📐 고등   🎓 대학   🌐 사회인
```

- **모델**: TinyLearnAI-30M (Decoder-only Transformer)
- **언어**: 한국어 + 영어 이중언어 BPE Tokenizer
- **Teacher**: Ollama 로컬 API (합성 데이터 생성 + 지식 증류)
- **환경**: Mac Apple Silicon (MPS) → Cloud CUDA 자동 전환
- **망각 방지**: EWC + Replay Buffer (Continual Learning)

---

## 🏗️ 모델 아키텍처

```
TinyLearnAI-30M
├── Token Embedding (32K vocab × 512 hidden)
├── 6 × TransformerBlock
│   ├── Pre-LayerNorm
│   ├── Multi-Head Causal Attention (8 heads, RoPE)
│   ├── Pre-LayerNorm
│   └── Feed-Forward (512 → 1024 → 512, GELU)
├── Final LayerNorm
└── LM Head (tied with embedding) ← 가중치 공유로 파라미터 절약
~29M 파라미터 | Mac MPS 메모리 <2GB
```

---

## 🚀 빠른 시작

### 1. 환경 설정

```bash
# 저장소 클론
cd EurekaAI

# 가상환경 (권장)
python -m venv .venv && source .venv/bin/activate

# 패키지 설치
pip install -r requirements.txt
```

### 2. Ollama Teacher 준비 (권장)

```bash
# Ollama 설치 후
ollama serve
ollama pull llama3.2:3b   # 또는 gemma3:4b
```

### 3. 토크나이저 학습 (최초 1회)

```bash
python scripts/setup_tokenizer.py
```

### 4. Stage 0 시작! 🍼

```bash
# 데이터 준비 + 학습
python run.py --stage 0

# 또는 데이터만 먼저 준비
python run.py --stage 0 --prep-only
```

### 5. 전체 커리큘럼 실행

```bash
# Stage 0 → 6 순차 실행
./scripts/run_all_stages.sh

# 특정 stage부터 재시작
./scripts/run_all_stages.sh 2
```

---

## 📁 디렉토리 구조

```
EurekaAI/
├── core/
│   ├── model/
│   │   ├── architecture.py     # TinyLearnAI-30M Transformer
│   │   ├── config.py           # 모델/학습 설정 (dataclass)
│   │   └── tokenizer_utils.py  # 한영 BPE Tokenizer
│   ├── teacher/
│   │   └── ollama_teacher.py   # Ollama API 연동
│   ├── curriculum/
│   │   ├── data_manager.py     # 커리큘럼 데이터 로딩
│   │   └── progression.py      # 단계 졸업 관리
│   ├── training/
│   │   ├── trainer.py          # 범용 Trainer (Mac/Cloud)
│   │   └── continual.py        # EWC + Replay Buffer
│   └── evaluation/
│       └── benchmarks.py       # 단계별 벤치마크
│
├── stages/
│   ├── stage0_newborn/         # 🍼 신생아
│   │   ├── config.yaml
│   │   ├── data_prep.py
│   │   └── train.py
│   ├── stage1_toddler/         # 🧸 유아
│   ├── stage2_elementary/      # 📚 초등
│   ├── stage3_middle/          # 🔢 중등
│   ├── stage4_high/            # 📐 고등
│   ├── stage5_university/      # 🎓 대학
│   └── stage6_social/          # 🌐 사회인
│
├── data/
│   ├── tokenizer/              # 토크나이저 모델
│   ├── raw/                    # 원본 데이터
│   └── processed/              # 전처리 완료 (.jsonl)
│       ├── stage0/
│       ├── stage1/
│       └── ...
│
├── checkpoints/                # 단계별 체크포인트
├── scripts/
│   ├── setup_tokenizer.py      # 토크나이저 초기 설정
│   └── run_all_stages.sh       # 전체 파이프라인 실행
│
├── run.py                      # 메인 진입점
├── docker-compose.yml
├── Dockerfile
└── requirements.txt
```

---

## 📊 단계별 커리큘럼

| Stage | 이름 | 학습 목표 | 데이터 | 예상 시간 | 졸업 기준 |
|-------|------|-----------|--------|-----------|-----------|
| 0 | 🍼 신생아 | 문자·패턴 학습 | ~1MB | ~30분 | PPL < 30 |
| 1 | 🧸 유아 | 단어·기초 문장 | ~10MB | ~30분 | Accuracy > 50% |
| 2 | 📚 초등 | 문법·기초 지식 | ~50MB | ~1시간 | F1 > 60% |
| 3 | 🔢 중학교 | 논리·추론 | ~100MB | ~2시간 | F1 > 65% |
| 4 | 📐 고등 | 비판적 사고 | ~200MB | ~2시간 | Acc > 45% |
| 5 | 🎓 대학 | 전문 지식 | ~500MB | ~3시간 | Acc > 40% |
| 6 | 🌐 사회인 | 통합 지능 | ~1GB | ~4시간 | Score > 4.0 |

---

## 🤖 Ollama Teacher 활용

| Stage | Teacher 역할 |
|-------|-------------|
| 0~1 | 유아어·동화 스타일 synthetic 데이터 생성 |
| 2~3 | QA Pair 생성, Chain-of-Thought 풀이 증류 |
| 4~5 | 논술 피드백(보상 신호), 전문 QA 생성 |
| 6 | A/B Preference 판별 → RLHF-lite |

```python
# 직접 사용 예시
from core.teacher.ollama_teacher import OllamaTeacher

teacher = OllamaTeacher(model="llama3.2:3b")
qa_pairs = teacher.generate_qa_pairs("태양은 지구에서 약 1억 5천만 km 떨어져 있습니다.", stage=2)
cot = teacher.get_chain_of_thought("x² - 5x + 6 = 0을 풀어라", stage=3)
score = teacher.score_response("수도는?", "서울입니다.", stage=2)
```

---

## ☁️ Cloud 이전 (Mac → GCP/AWS)

```bash
# Mac에서 동일하게 실행됨 (DEVICE=auto)
# Cloud에서는 CUDA 자동 감지
DEVICE=cuda python run.py --stage 0

# Docker로 완전 이식
docker-compose up trainer

# 더 큰 모델로 확장
# config.py의 small_config() 또는 medium_config() 사용
```

---

## 🔬 현재 상태 확인

```bash
python run.py --status
```

```
══════════════════════════════════════════════════════════
   EurekaAI — Curriculum Progress
══════════════════════════════════════════════════════════
  🎓 Stage 0: 🍼 신생아 (Newborn)          [graduated]  PPL=24.3
  ✅ Stage 1: 🧸 유아기 (Toddler)           [completed]
  🔄 Stage 2: 📚 초등학교 (Elementary)      [training]
  ⏳ Stage 3: 🔢 중학교 (Middle School)     [pending]
  ...
══════════════════════════════════════════════════════════
```

---

## 📈 Continual Learning 전략

```
Stage N 완료  →  Fisher Matrix 계산 (EWC)
                  ↓
Stage N 데이터 일부 → Replay Buffer 저장
                  ↓
Stage N+1 학습  →  Task Loss + EWC Penalty + Replay Data (20%)
```

---

## 🛠 개발 가이드

```bash
# 특정 stage 평가만
python run.py --eval --stage 0

# 커스텀 설정으로 학습
python run.py --stage 0 --config stages/stage0_newborn/config.yaml

# 테스트
pytest tests/ -v
```

---

## 📄 라이선스

MIT License — 자유롭게 사용, 수정, 배포 가능합니다.

---

*"유레카! 모든 지식은 스스로 발견될 때 가장 빛난다."* ✨
