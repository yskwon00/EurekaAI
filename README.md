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
├── s## 📊 단계별 커리큘럼

| Stage | 이름 | 학습 목표 | Teacher 역할 | 졸업 기준 |
|-------|------|-----------|-------------|---------|
| 0 | 🍼 신생아 | 문자·패턴 학습 | ❌ 미사용 | PPL ≤ 30 |
| 1 | 🧸 유아기 | 단어·기초 문장 | ❌ 미사용 | PPL ≤ 20 |
| 2 | 📚 초등학교 | 문법·기초 지식 | **Q&A 쌍 생성** | PPL ≤ 15 |
| 3 | 🔢 중학교 | 논리·추론 | **Chain-of-Thought 풀이** | PPL ≤ 10 |
| 4 | 📐 고등학교 | 비판적 사고 | **채점 필터링** (≥0.6) | PPL ≤ 8 |
| 5 | 🎓 대학교 | 전문 지식 | **고품질 필터** (≥0.7) | PPL ≤ 6 |
| 6 | 🌐 사회인 | 통합 지능 | **RLHF 선호 쌍** | PPL ≤ 5 |

---

## 🗂️ 단계별 데이터 생성 상세

### Stage 0~1 (신생아·유아기) — Teacher 미사용

```
[신생아] Seed 텍스트(단어·모음·숫자) × 20회 반복
         + TinyStories 한국어 번역본
         + Korean Wikipedia (5단어 이하 문장)
         총 ~120,000건 | max_seq_len: 128

[유아기]  Seed 문장(일상표현 한영) × 10회
         + TinyStories 원본 (동화책 스타일)      78%
         + Korean Wikipedia (중간 난이도 문장)   17%
         + Stage 0 Replay (망각 방지)            5%
         총 ~70,000건 | max_seq_len: 256
```

---

### Stage 2 (초등학교) — Teacher Q&A 생성 시작

```
목표: 기초 지식을 질문-답변 형식으로 학습

[Teacher 역할]
  1. 한국어 Wikipedia 150개 문서 로드
  2. 각 문서 → Ollama에게 초등 수준 Q&A 3쌍 생성 요청
  3. JSON 파싱 → 저장 (캐시: data/teacher_cache/)

생성 형식:
  Q: 광합성이란 무엇인가요?
  A: 식물이 햇빛, 물, 이산화탄소로 음식을 만드는 과정이에요.

데이터 구성 (~70,000건):
  TinyStories (영문 기초)       78%
  Korean Wikipedia 문장         17%
  Stage 0+1 Replay              5%
  Teacher Q&A + Synthetic       <1% (캐시 활용)

속도 최적화:
  - ThreadPoolExecutor(max_workers=2) 병렬 처리
  - OllamaTeacher 캐시 (MD5 해시 기반) — 재실행 시 즉시 반환
  - timeout=240초 (로컬 Ollama 과부하 방지)
```

---

### Stage 3 (중학교) — Chain-of-Thought 핵심 ⭐

```
목표: 단계적 논리 추론 능력 학습

[Teacher 역할: get_chain_of_thought()]
  수학 문제 (10개):
    - "2x + 5 = 13일 때 x를 구하세요."
    - "삼각형의 넓이 계산 (밑변 6, 높이 4)"
    - "1부터 100까지의 합"
    - "원의 둘레 (반지름 7cm)" 등

  과학 문제 (8개):
    - "광합성 과정을 단계별로 설명해주세요."
    - "뉴턴의 운동 법칙 3가지"
    - "물의 상태 변화" 등

  논리 문제 (5개): (영문)
    - "All cats are animals... Do cats breathe? Explain."
    - "3L/5L 저그로 4L 측정하기"
    - 수열 패턴 추론 등

  → 각 문제당 Teacher가 단계별 풀이(CoT) 생성
  → 문제당 8회 반복 = ~184 × 8 = 1,472개 CoT 샘플
  → ThreadPoolExecutor(max_workers=2) 병렬 처리

생성 형식:
  문제: 2x + 5 = 13일 때 x를 구하세요.
  
  풀이:
  1단계: 식의 양변에서 5를 빼면 2x = 8이 됩니다.
  2단계: 양변을 2로 나누면 x = 4가 됩니다.
  3단계: 검증: 2(4) + 5 = 13 ✓

데이터 구성:
  Teacher CoT 풀이               ~30%
  Korean Wikipedia 단락 (80~400자) ~50%
  Stage 0~2 Replay (10%)        ~15%
  Ollama 합성 데이터              ~5%

max_seq_len: 512
```

---

### Stage 4 (고등학교) — 품질 채점 필터링 ⭐

```
목표: 비판적 사고 + 고품질 데이터로만 학습

[Teacher 역할 1: generate_qa_pairs() + score_response()]
  1. Wikipedia 100개 문서 → 고등 수준 Q&A 3쌍 생성
  2. 생성된 각 답변을 score_response()로 채점 (0.0~1.0)
  3. 점수 ≥ 0.6인 답변만 학습 데이터로 채택 ← 품질 필터!

  예시:
    Q: 민주주의의 한계는 무엇인가요?
    A: (Teacher 채점 0.8 → 채택)
    A: (Teacher 채점 0.4 → 제외)

[Teacher 역할 2: 에세이 생성]
  한국어 주제 (8개):
    - "환경 보호를 위해 우리가 할 수 있는 일들을 논술하세요."
    - "인터넷이 사회에 미치는 긍정적, 부정적 영향을 분석하세요."
    - "청소년 스마트폰 과사용 문제와 해결 방안" 등

  영어 주제 (7개):
    - "Discuss the impact of social media on modern communication."
    - "Should school uniforms be mandatory? Argue both sides." 등

  → 각 주제 × 5회 반복 = 75개 에세이 생성
  → Teacher가 고등학생 수준으로 작성

생성 형식 (에세이):
  주제: 환경 보호를 위해 우리가 할 수 있는 일들을 논술하세요.
  
  환경 보호는 현대 사회에서 가장 중요한 과제 중 하나입니다.
  첫째, 일상에서 에너지 절약을 실천할 수 있습니다...

데이터 구성:
  채점 Q&A (score ≥ 0.6)         ~20%
  Teacher 에세이                  ~15%
  Korean Wikipedia 단락 (100~600자) ~45%
  Stage 0~3 Replay (10%)         ~20%

max_seq_len: 1024
```

---

### Stage 5~6 (대학교·사회인) — RLHF 적용

```
[Stage 5 대학교]
  - 학술 Q&A + score_response() ≥ 0.7 고품질 필터링
  - Wikipedia 학술 단락 (150~800자)
  - 이전 단계 리플레이 8%
  - max_seq_len: 1024

[Stage 6 사회인]
  - create_preference_pairs(): 두 답변 생성 → Teacher가 더 나은 것 선택 (RLHF)
  - 대화 형식 데이터 (User/Assistant)
  - 코드 예제 (Python 패턴, 알고리즘)
  - max_seq_len: 2048
```

---

## 🤖 Ollama Teacher 활용

```python
from core.teacher.ollama_teacher import OllamaTeacher

teacher = OllamaTeacher(model="gemma4:e4b", timeout=240.0)

# Stage 2: Q&A 쌍 생성
qa_pairs = teacher.generate_qa_pairs(passage="광합성은...", stage=2, n=3)

# Stage 3: Chain-of-Thought 단계별 풀이
cot = teacher.get_chain_of_thought("2x + 5 = 13을 풀어라", stage=3)

# Stage 4~5: 답변 품질 채점 (0.0~1.0)
score = teacher.score_response(question="민주주의란?", answer="...", stage=4)

# Stage 6: 선호 쌍 생성 (RLHF)
pref = teacher.create_preference_pairs(question="AI란?", answer_a="...", answer_b="...", stage=6)
# → {"chosen": "...", "rejected": "...", "reason": "..."}
```

> **캐시**: 모든 Teacher 응답은 `data/teacher_cache/`에 MD5 해시로 캐시됨.
> 재실행 시 즉시 반환되어 중단 후 재시작 비용 최소화.

---

## 📈 Continual Learning 전략

```
Stage N 완료  →  Fisher Matrix 계산 (EWC)
                  ↓
Stage N 데이터 일부 → Replay Buffer 저장
                  ↓
Stage N+1 학습  →  Task Loss + EWC Penalty + Replay Data

EWC λ 감소 추이:
  Stage 1: λ=5000 (Stage 0 지식 강하게 보존)
  Stage 2: λ=3000
  Stage 3: λ=2000
  Stage 4: λ=1500
  Stage 5: λ=1000
  Stage 6: λ=500  (새 학습에 집중)

Replay 비율:
  Stage 1~2: 15~20%  Stage 3~4: 10%  Stage 5~6: 5~8%
```

---

## 🔬 현재 상태 확인

```bash
python run.py --status
```

```
============================================================
   EurekaAI — Curriculum Progress
============================================================
  ✅ Stage 0: 🍼 신생아 (Newborn)                [completed]
      best_metric=1.0795, steps=12000
  ✅ Stage 1: 🧸 유아기 (Toddler)                [completed]
      best_metric=1.1000, steps=2200
  🔄 Stage 2: 📚 초등학교 (Elementary)            [training]
  ⏳ Stage 3: 🔢 중학교 (Middle School)          [pending]
  ...
============================================================
```

**로그 실시간 확인:**
```bash
# 학습 로그
tail -f logs/stage2_elementary_*.log | grep -E "Step|Eval"

# 데이터 준비 로그
tail -f logs/stage2_data_prep_*.log | grep -E "완료|✅|저장"
```

---

## 🛠 개발 가이드

```bash
# 특정 stage 학습
python stages/stage2_elementary/train.py

# 데이터만 준비
python stages/stage3_middle/data_prep.py

# 백그라운드 학습
nohup .venv/bin/python3 stages/stage2_elementary/train.py > /dev/null 2>&1 &

# 학습 진행 확인
tail -f logs/stage2_elementary_*.log

# Eval PPL만 확인
grep "📊 Eval" logs/stage2_elementary_*.log
```

---

*"유레카! 모든 지식은 스스로 발견될 때 가장 빛난다."* ✨

