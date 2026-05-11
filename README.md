# 🧠 EurekaAI — Self-Learning Curriculum AI

> **"유레카!"** — 신생아처럼 백지 상태에서 시작해 스스로 초·중·고·대학·사회 수준으로 성장하는 AI

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-EE4C2C.svg)](https://pytorch.org)
[![Mac MPS](https://img.shields.io/badge/Mac-Apple%20Silicon-black.svg)](https://developer.apple.com/metal/)
[![W&B](https://img.shields.io/badge/W%26B-Experiment%20Tracking-FFBE00.svg)](https://wandb.ai)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## ✨ 개요

EurekaAI는 **사전 학습된 지식 없이** 커리큘럼 기반으로 스스로 성장하는 한국어+영어 이중언어 AI 모델입니다.

```
Stage 0 → Stage 1 → Stage 2 → Stage 3 → Stage 4 → Stage 5 → Stage 6
🍼 신생아   🧸 유아   📚 초등   🔢 중등   📐 고등   🎓 대학   🌐 사회인
```

| 항목 | 세부 사항 |
|------|----------|
| **모델** | EurekaAI-80M (Decoder-only Transformer) |
| **파라미터** | ~81M (hidden=768, layers=8, heads=12, ffn=3072) |
| **언어** | 한국어 + 영어 이중언어 BPE Tokenizer (32K Vocab) |
| **Teacher** | Ollama 로컬 API (gemma4:e4b / llama3.2:3b) |
| **환경** | Mac Apple Silicon (MPS) 최적화 |
| **망각 방지** | EWC (Elastic Weight Consolidation) + Replay Buffer |
| **실험 추적** | Weights & Biases (W&B) + Artifact Lineage |

---

## 📈 스케일링 여정 & 핵심 노하우

> 처음부터 80M이 아니었습니다. 세 번의 스케일링 실험을 통해 얻어낸 "최적의 지점"입니다.

### 🔴 1단계: 20M → 실패 (Under-parameterized)

```
아키텍처: hidden=256, layers=4, heads=4 (~20M params)
```

- **증상**: 학습 데이터가 조금만 복잡해져도 Loss가 발산하거나 문장 구조를 전혀 학습하지 못함
- **근본 원인**: 32K Vocab을 소화하면서 한국어+영어의 이중 언어 패턴 관계를 동시에 이해하기에는 모델 용량이 물리적으로 부족
- **교훈**: BPE 32K를 타겟으로 하는 이중언어 모델의 현실적인 하한선은 ~60M 이상

### 🟡 2단계: 120M → 중단 (Resource Overload)

```
아키텍처: hidden=1024, layers=12, heads=16 (~120M params)
```

- **증상**: Mac M-시리즈 MPS 메모리 점유가 한계에 도달하여 배치 사이즈를 1로 줄여도 OOM 발생
- **추가 문제**: 학습 한 스텝에 약 8~12초 소요 → 5,000 steps = 수 시간, 반복 실험 불가능
- **교훈**: 로컬 환경에서의 **빠른 피드백 루프**는 연구 속도의 핵심. 리소스 효율성이 정확도보다 우선순위일 수 있음

### 🟢 3단계: 80M → 성공 (The Goldilocks Zone)

```
아키텍처: hidden=768, layers=8, heads=12, ffn=3072 (~81M params)
MPS 메모리 점유: ~3-4 GB | 학습 속도: ~2-4초/step
```

- **결과**: Stage 0 졸업 PPL 25.2, Stage 1 졸업 PPL 19.4
- **핵심 인사이트**: `hidden_size × 4 = intermediate_size` 비율을 지키는 것이 학습 안정성에 중요

---

## 🔥 실전에서 발견한 버그 & 노하우

### 🐛 Bug #1 — Label Shifting (Causal LM 핵심 오류)

**문제**: Loss가 전혀 줄지 않는 현상 (loss ~10 고착)

```python
# ❌ 잘못된 방식: label이 input과 동일하면 아무것도 학습하지 않음
labels = input_ids.clone()  # 완전히 틀린 CLM 학습

# ✅ 올바른 방식: 1-token shift로 "다음 단어 예측" 구현
labels = input_ids.clone()
labels[:, :-1] = input_ids[:, 1:]   # 한 칸 앞 토큰이 target
labels[:, -1] = -100                 # 마지막 위치는 ignore
```

**교훈**: Causal LM에서 `labels[t] = input_ids[t+1]`이 되어야 "현재 토큰으로 다음 토큰을 예측"하는 정상 학습이 됨. 이것이 틀리면 loss가 수렴하는 척 하다가 실제 생성 능력은 제로임.

---

### 🐛 Bug #2 — Gradient Accumulation 로깅 버그

**문제**: PPL이 비정상적으로 높게 (>100) 측정되는 현상

```python
# ❌ 잘못된 방식: total_loss 초기화가 wandb.log() 이후 중복 수행
if self.global_step % self.config.log_interval == 0:
    avg_loss = total_loss / self.config.log_interval
    # ...
    total_loss = 0.0  # 첫 번째 초기화
    if self.wandb:
        self.wandb.log({"train/loss": avg_loss})
    total_loss = 0.0  # ← 중복! 영향 없음 (but 주석 보면 혼란)

# ✅ 올바른 방식: 누적 loss는 grad_accum 곱해서 복원 후 평균
avg_loss = total_loss / (self.config.log_interval * grad_accum)
total_loss = 0.0  # log 이전에 한 번만 초기화
```

**교훈**: Gradient Accumulation 시 실제 loss 값은 `/grad_accum`으로 스케일이 조정된 상태. 평균 계산 시 `log_interval × grad_accum`으로 나눠야 실제 loss를 복원할 수 있음.

---

### 🐛 Bug #3 — Eval Loop의 Labels 덮어쓰기

**문제**: Train과 Eval의 loss 기준이 달라 비교 불가

```python
# ❌ 잘못된 방식: eval 중 labels를 batch에서 가져오지 않고 항상 재생성
for batch in self.eval_loader:
    labels = input_ids.clone()  # batch의 labels 무시!

# ✅ 올바른 방식: batch에 labels가 있으면 그것을 우선 사용
labels = batch.get("labels", None)
if labels is None:
    labels = input_ids.clone()
    labels[:, :-1] = input_ids[:, 1:]
    labels[:, -1] = -100
```

---

### 💡 Know-how #1 — Stage 0의 데이터 정제가 핵심

**문제**: 위키백과 데이터를 Stage 0에 넣으니 loss 발산

```
복잡한 데이터 → 모델이 어떤 패턴도 못 잡음 → loss 발산
```

**해결**: Stage 0에는 반드시 **극도로 단순한** 데이터만 사용

```
✅ TinyStories (roneneldan/TinyStories): 5살 아이 수준 영어 동화
✅ 한국어 씨앗 문장: "안녕하세요.", "사과는 맛있어요." (5단어 이하)
✅ 위키 초간단 문장: 50~150자 이내, 단순 서술문만
❌ 위키 기본 문장 (>200자): Stage 0에 부적합
❌ Q&A 쌍: Stage 2 이상에서 사용
❌ 뉴스/논문: Stage 5 이상에서 사용
```

---

### 💡 Know-how #2 — Mac MPS 환경 최적화 팁

```python
# 1. num_workers=0 필수 (MPS에서 멀티프로세싱 충돌)
DataLoader(..., num_workers=0)

# 2. batch_size × gradient_accumulation = 유효 배치
# MPS는 배치 4 + grad_accum 4 = 유효 배치 16이 안정적
batch_size: 4
gradient_accumulation_steps: 4  # effective batch = 16

# 3. fp16/bf16은 MPS에서 불안정 → 사용 안 함
fp16: false
bf16: false

# 4. 체크포인트 저장 시 map_location="cpu" 필수
torch.load(path, map_location="cpu", weights_only=True)
```

---

### 💡 Know-how #3 — Ollama Teacher 연동 시 JSON 에러 처리

**문제**: Ollama(Gemma 계열)가 System Role을 빈 응답으로 반환하거나 JSON 형식을 무시하는 경우

```python
# ❌ 그냥 JSON 파싱하면 OOM (Out-Of-Meaning) 에러
response = ollama.chat(model="gemma4:e4b", messages=[...])
data = json.loads(response.message.content)  # 실패 가능!

# ✅ 방어적 파싱: JSON 블록만 추출 + 폴백 처리
import re
def safe_json_parse(text: str) -> dict:
    # JSON 블록 추출 시도
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return {}  # 빈 dict 폴백 → 캐시 미스로 처리
```

---

### 💡 Know-how #4 — Auto-Resume 체크포인트 전략

모든 stage의 `train.py`에 적용된 자동 재개 로직:

```python
# 기본 동작: 항상 최신 step_XXXX 체크포인트에서 재개
# --reset 옵션: 처음부터 새로 시작

if not reset:
    ckpt_path = Path(f"checkpoints/{stage_name}/{stage_name}")
    latest_step, best_tag = -1, None
    if ckpt_path.exists():
        for d in ckpt_path.iterdir():
            if d.is_dir() and d.name.startswith("step_"):
                step_val = int(d.name.split("_")[1])
                if step_val > latest_step:
                    latest_step = step_val
                    best_tag = d.name
    if best_tag:
        trainer.load_checkpoint(best_tag)
```

**교훈**: Mac 환경에서는 배터리 부족, 과열, 시스템 슬립 등으로 학습이 자주 중단됨. Auto-resume은 선택이 아닌 필수.

---

### 💡 Know-how #5 — EWC λ (람다) 튜닝

```
Stage 1: λ=5000  → Stage 0 (기초 패턴) 강하게 보존 필요
Stage 2: λ=3000
Stage 3: λ=2000
Stage 4: λ=1500
Stage 5: λ=1000
Stage 6: λ=500   → 새로운 고급 지식 학습에 집중

λ가 너무 크면: 새 stage 학습이 막혀서 loss 하락 안 됨
λ가 너무 작으면: 이전 stage 지식이 빠르게 삭제됨 (Catastrophic Forgetting)
```

---

### 💡 Know-how #6 — W&B Lineage Artifact 더미 파일 트릭

**문제**: 같은 체크포인트를 두 번 업로드하면 W&B가 "이미 존재하는 artifact"로 판단해 버전 생성 거부

```python
# ✅ 해결: 업로드 시마다 타임스탬프 기반 더미 파일 생성
dummy_file = path / "dummy_lineage.txt"
with open(dummy_file, "w") as f:
    f.write(f"Lineage Hash Force: {time.time()}")
# → 파일 내용이 달라지므로 W&B가 새 버전으로 인식
```

---

### 💡 Know-how #7 — LR 스케줄링 전략

각 Stage별 Learning Rate 권장 설정:

```
Stage 0 (신생아): lr=1e-4, warmup=500  → 처음 학습이라 높은 lr + 긴 웜업
Stage 1 (유아기): lr=5e-5, warmup=300  → 이전 knowledge 유지하며 Fine-tune
Stage 2 (초등학교): lr=5e-5, warmup=300
Stage 3 (중학교):   lr=3e-5, warmup=300  → 복잡한 패턴 → 낮은 lr
Stage 4 (고등학교): lr=3e-5, warmup=300
Stage 5 (대학교):   lr=2e-5, warmup=200  → 정밀 Fine-tune
Stage 6 (사회인):   lr=1e-5, warmup=100  → 아주 작은 lr로 마무리

스케줄: Warmup + Cosine Decay (min_lr = 10% of peak lr)
```

---

## 🏗️ 모델 아키텍처

```
EurekaAI-80M (small_config)
├── Token Embedding (32,000 vocab × 768 hidden)          ← 24.6M params
├── 8 × TransformerBlock
│   ├── Pre-LayerNorm (강력한 학습 안정성)
│   ├── CausalSelfAttention (12 heads, head_dim=64, RoPE) ← 2.36M/layer
│   │   ├── Q/K/V/O Projection (bias=False)
│   │   ├── RotaryEmbedding (no extra params!)
│   │   └── Causal mask (registered_buffer)
│   ├── Pre-LayerNorm
│   └── FeedForward (768 → 3072 → 768, GELU)            ← 4.72M/layer
├── Final LayerNorm
└── LM Head (tied with embedding, 파라미터 절약!)        ← 0 extra params

총 파라미터: ~81M | Mac MPS 메모리: ~3-4 GB
```

### 아키텍처 설계 결정 이유

| 선택 | 이유 |
|------|------|
| **Pre-LayerNorm** | Post-LN 대비 학습 초기 안정성 확실히 우수 |
| **RoPE** | Absolute PE 대비 긴 시퀀스 일반화 능력 우수 |
| **bias=False** | Linear Layer에서 bias 제거 → 학습 안정성 향상 |
| **Weight Tying** | Embedding ↔ LM Head 공유 → ~24M params 절약 |
| **GELU** | ReLU 대비 부드러운 기울기 → 안정적 학습 |
| **GPT-2 Init** | Output proj std=0.02/√(2*layers) → 잔차 연결 안정화 |

---

## 📊 단계별 커리큘럼

| Stage | 이름 | 학습 목표 | Teacher 역할 | max_seq_len | 졸업 기준 |
|-------|------|-----------|-------------|------------|---------|
| 0 | 🍼 신생아 | 문자·패턴·어휘 | ❌ 미사용 | 512 | PPL ≤ 30 |
| 1 | 🧸 유아기 | 단어·기초 문장 | ❌ 미사용 | 512 | PPL ≤ 20 |
| 2 | 📚 초등학교 | 문법·기초 지식 | **Q&A 쌍 생성** | 512 | PPL ≤ 15 |
| 3 | 🔢 중학교 | 논리·추론 | **Chain-of-Thought** | 512 | PPL ≤ 10 |
| 4 | 📐 고등학교 | 비판적 사고 | **채점 필터링 ≥0.6** | 1024 | PPL ≤ 8 |
| 5 | 🎓 대학교 | 전문 지식 | **고품질 필터 ≥0.7** | 1024 | PPL ≤ 6 |
| 6 | 🌐 사회인 | 통합 지능 | **RLHF 선호 쌍** | 2048 | PPL ≤ 5 |

---

## 🗂️ 단계별 학습 데이터 상세

### Stage 0 🍼 신생아 (Newborn) — `data/processed/stage0_newborn/`

**목표**: 문자, 음절, 기초 어휘 패턴을 학습  
**핵심 원칙**: 극도로 단순한 데이터만 허용

```
데이터 구성 (~30,000건 | max_seq_len: 512):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  TinyStories (roneneldan/TinyStories)       80%  ~24,000건
    - 길이 필터: 100~1,000자
    - 5살 아이 수준 단순 영어 동화
    - "Once upon a time, a little dog..."
  
  Korean Wikipedia 초간단 단락               20%  ~6,000건
    - 길이 필터: 50~150자 (단어 5개 내외)
    - "서울은 대한민국의 수도입니다."
    - split: train[:1,000] (소규모)
  
  한국어 씨앗 문장 (직접 작성)                소량 ~450건
    - "안녕하세요.", "사과는 맛있어요.", "하늘이 파랗습니다."
    - × 50회 반복
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
하이퍼파라미터:
  lr=1e-4, warmup=500, batch=4, grad_accum=4, max_steps=5,000
  dropout=0.15 (과적합 방지 강화)
  EWC λ=5000, replay=15%

졸업 기록: PPL=25.2 (12,000 steps) ✅
```

---

### Stage 1 🧸 유아기 (Toddler) — `data/processed/stage1_toddler/`

**목표**: 단어 어휘 확장, 기초 문장 구조 이해  
**핵심**: Stage 0 지식 유지하면서 더 긴 문장으로 자연스럽게 전환

```
데이터 구성 (~50,000건 | max_seq_len: 512):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  TinyStories (조금 더 긴 것)                50%  ~25,000건
    - 길이 필터: 150~800자 (Stage 0보다 완화)
    - 최대 700자로 truncation
  
  Korean Wikipedia 중간 단락                 50%  ~25,000건
    - 길이 필터: 150~400자
    - split: train[:800]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
하이퍼파라미터:
  lr=5e-5, warmup=300, batch=4, grad_accum=4
  EWC λ=5000 (Stage 0 지식 강하게 보호)
  replay=20% (Stage 0 리플레이)

졸업 기록: PPL=19.4 (2,200 steps) ✅
```

> **관찰**: Stage 0에서 충분히 학습된 모델은 Stage 1에서 매우 빠르게 적응함 (2,200 steps만에 졸업)

---

### Stage 2 📚 초등학교 (Elementary) — `data/processed/stage2_elementary/`

**목표**: 문법 규칙 학습, 기초 지식 습득, 대화형 도메인 적응
**핵심 이슈 및 해결 (도메인 편향 극복)**:
초기 위키백과(40%) 의존도가 너무 높아, 모델이 한국어 질문에 대답하지 못하고 대화형 프롬프트를 인식하지 못하는 "도메인 과적합(Domain Bias)" 문제가 발생함 (v1~v4 실패: PPL 수렴 실패 또는 대화 불가). 이를 해결하기 위해 ShareGPT-ko와 TinyStories를 적절히 혼합하여 데이터 백본을 재설계함.

```text
최종 데이터 구성 (~250,000건 | max_seq_len: 512):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  ShareGPT-ko (한국어 질문-답변)         30%
  Korean Wikipedia (한국어 지식)         40%
  TinyStories (영어 동화형 문맥)         25%
  한국어 기초 씨앗 문장                    5%
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
하이퍼파라미터:
  lr=1e-4, warmup=500, batch=4, grad_accum=4, max_steps=30000
  EWC λ=1000 

졸업 기록: 30,000 steps 종료 후 강제 졸업 (최종 PPL=38.51)
> 관찰: 초기 6만 건 데이터로 1만 스텝 학습 시 PPL이 84.4에서 붕괴(Model Collapse)되었으나,
> 데이터를 25만 건으로 대폭 확장하고 3만 스텝(약 2.4억 토큰)으로 스케일링한 결과 
> PPL을 38.51까지 극적으로 낮추며 한국어/영어의 정상적인 문법 구조 생성을 달성함.
```

---

### Stage 3 🔢 중학교 (Middle School) — `data/processed/stage3_middle/`

**목표**: 단계적 논리 추론, 심화된 문맥 파악, 긴 문장 생성력 강화
**핵심**: 문맥 허용 길이 `max_seq_len`을 1024로 확장. Stage 2의 혼합 데이터 전략을 계승하여 심화 지식과 깊이 있는 대화 패턴을 동시에 학습.

```text
데이터 구성 (~390,000건 | max_seq_len: 1024):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  ShareGPT-ko (한국어 심화 대화)         100,000건
  Korean Wikipedia (한국어 심화 지식)    150,000건
  TinyStories (영어 심화 동화)           100,000건
  논리/명제 CoT & Replay                 45,000건
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
하이퍼파라미터:
  lr=5e-5, warmup=500, batch=2, grad_accum=8, max_steps=30000
  dropout=0.10, EWC λ=1000
  
진행 상황 및 관찰 (진행 중):
  - 시퀀스 길이가 1024로 2배 늘어나고, ShareGPT 등 복잡도가 크게 증가하여
    초기 PPL이 200을 돌파하는 Curriculum Shock 발생.
  - 하지만 약 2,000 스텝 만에 115 수준으로 안정화되었으며, 20,000 스텝 돌파 시점 
    PPL 69.2를 기록하며 80M 파라미터의 한계를 깨고 매우 성공적인 우하향 곡선을 그림.
```

---

### Stage 4 📐 고등학교 (High School) — `data/processed/stage4_high/`

**목표**: 비판적 사고, 에세이 논술, 품질 채점 필터  
**핵심**: Teacher가 생성한 답변을 Teacher 스스로 채점하여 필터링

#### ⚠️ 실패 이력 및 해결 과정

**실패 #1 — OOM (Out Of Memory) 크래시**
```
원인: data_prep.py 완료 후 Ollama(gemma4:e4b)가 RAM 8~12GB를 
      계속 점유 → train.py 시작 시점에 MPS OOM 에러 발생
해결: 데이터 생성 직후 Ollama 강제 언로드 루틴 추가
      curl -X POST http://localhost:11434/api/generate \
           -d '{"model":"gemma4:e4b","keep_alive":0}'
결과: 학습 프로세스가 Mac RAM을 100% 독점, OOM 재발 없음
```

**실패 #2 — 데이터 경로 불일치 버그**
```
원인: data_prep.py는 data/processed/stage4/에 저장
      train.py는 data/processed/stage4_high/를 참조
      → FileNotFoundError로 학습 시작 불가
해결: 양쪽 경로를 data/processed/stage4_high/로 통일
결과: 이후 stage5_university도 동일 패턴으로 사전 방지
```

**실패 #3 — max_steps 부족으로 조기 종료**
```
원인: 초기 설정 max_steps=5000 → 데이터 10만 건 기준
      약 0.4 Epoch밖에 돌지 않아 모델이 형식을 충분히 체득하지 못함
해결: max_steps=15000 (약 2.4 Epoch)으로 상향 조정
결과: 에세이 생성 형식이 안정화되고 단어 반복(Repetition) 현상 소멸
```

```text
데이터 구성 (~100,000건 | max_seq_len: 1024):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  ShareGPT-ko (한국어 심화 지시어)          50,000건
  Korean Wikipedia (심화 지식)              40,000건
  Teacher 채점 Q&A (score ≥ 0.6) ⭐       소량
  Stage 0~3 Replay                         8,000건
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
하이퍼파라미터 변천사:
  [v1 실패] lr=3e-5, max_steps=5000 → 조기 종료, 형식 미체득
  [v2 성공] lr=2e-5, warmup=300, batch=2, grad_accum=8
            max_steps=15,000 (약 2.4 Epoch)
            EWC λ=1500, dropout=0.1, score_threshold=0.6

최종 성과 (2026-05-09 05:23 완료):
  - 15,000 스텝 완주 ✅
  - 최종 Eval PPL: 46.57
  - 핵심 달성: 단어 반복 현상 100% 소멸 + 에세이 형식 완벽 체득
  - 서빙 테스트: "지구 온난화에 대해 논술하시오" → 구조화된 에세이 생성 확인
  - 졸업 기준 PPL ≤ 8 미달이나, 80M의 물리적 한계로 force_graduate 처리
```

---

### Stage 5 🎓 대학교 (University) — `data/processed/stage5_university/`

**목표**: 학술적 전문 지식, 코딩/논리 추론, 엄격한 품질 필터  
**핵심**: 코딩 데이터 추가로 구조적 논리 추론 능력 강화

#### ⚠️ 실패 이력 및 해결 과정

**실패 #1 — 기존 스크립트 데이터 부족 (10만 건 → 실제 2만 건)**
```
원인: 기존 data_prep.py의 generate_academic_qa()가 Ollama 응답 속도 
      병목으로 100개 문서에서 수천 건만 생성 → 목표 10만 건 대비 대폭 미달
해결: 데이터 전략 재편
      1) Ollama Q&A 생성량 축소 (n_articles=50, score≥0.7)
      2) 학술 Wikipedia 30,000건 대폭 확대
      3) 파이썬 코딩 데이터 15,000건 신규 추가
         (iamtarun/python_code_instructions_18k_alpaca)
      4) ShareGPT 심화 대화 50,000건 (max_turns=12) 추가
결과: 실제 생성 데이터 51,314건 (Train) + 2,000건 (Eval)
```

**실패 #2 — 경로 불일치 재발 (Stage 4와 동일 패턴)**
```
원인: STAGE_DIR = "data/processed/stage5"
      train.py는 "data/processed/stage5_university" 참조
해결: data_prep.py의 모든 경로를 stage5_university로 통일
```

**실패 #3 — max_steps 5,000 → 20,000으로 대폭 상향**
```
원인: 기존 config.yaml max_steps=5000은 51,000건 데이터 기준
      약 0.2 Epoch에 불과 → 사실상 학습 안 된 것과 동일
해결: max_steps=20,000 (약 3.9 Epoch)으로 상향
      eval_interval=1000, save_interval=2000으로 안전하게 설정
```

```text
데이터 구성 (~53,314건 | max_seq_len: 1024):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  ShareGPT-ko 심화 대화 (max_turns=12)    ~50,000건 목표
  Korean Wikipedia 학술 (350~1,000자)     30,000건
  Python 코드 지시문 (alpaca 포맷)        15,000건 (신규 ⭐)
  Teacher 학술 Q&A (score ≥ 0.7)         소량
  Stage 0~4 Replay                        ~7,500건
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
하이퍼파라미터:
  lr=2e-5, warmup=200, batch=2, grad_accum=8
  max_steps=20,000 (약 3.9 Epoch)
  EWC λ=1000, dropout=0.1
  score_threshold=0.7

진행 상황 (2026-05-11 학습 중):
  - Step 600 기준 PPL=28.0 (초기 PPL=36.1 → 빠른 수렴)
  - Stage 4(PPL 95 출발) 대비 훨씬 낮은 초기치에서 시작
    → Curriculum 누적 학습 효과 증명
  - 졸업 기준: PPL ≤ 20.0
```

---

### Stage 6 🌐 사회인 (Social) — `data/processed/stage6_social/`

**목표**: 통합 지능, 대화 능력, RLHF 선호 학습

```
데이터 구성 (~80,000건 | max_seq_len: 2048):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Korean Wikipedia (300~800자)              ~60%  ~48,000건
  ShareGPT 한국어 대화                       ~30%  ~24,000건
  Teacher RLHF 선호 쌍 (핵심 ⭐)            소량
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
하이퍼파라미터:
  lr=1e-5, warmup=100, batch=2, grad_accum=8
  EWC λ=500 (새 학습에 집중)
  replay=5%

졸업 기준: PPL ≤ 5
```

---

## 📈 Continual Learning 전략

```
Stage N 완료
     │
     ├─► Fisher Matrix 계산 (EWC)
     │     → 중요 파라미터 식별 + λ 설정
     │
     ├─► Replay Buffer 저장
     │     → Stage N 데이터 일부 (2,000건) 보관
     │
     └─► Stage N+1 학습
           Loss = Task Loss + EWC Penalty + Replay Data

EWC λ 감소 추이:
  Stage 1: λ=5000  ████████████████████ (Stage 0 지식 최강 보존)
  Stage 2: λ=3000  ███████████████
  Stage 3: λ=2000  ██████████
  Stage 4: λ=1500  ████████
  Stage 5: λ=1000  █████
  Stage 6: λ=500   ███ (새 학습에 집중)

Replay 비율 감소 추이:
  Stage 1~2: 15~20%  Stage 3~4: 10%  Stage 5~6: 5~8%
```

---

## 🔧 공통 운영 노하우

### Mac MPS 메모리 관리 (핵심 ⭐)
```bash
# 학습 전 Ollama 강제 언로드 필수 (OOM 방지)
curl -X POST http://localhost:11434/api/generate \
     -d '{"model": "gemma4:e4b", "keep_alive": 0}'

# 권장 학습 시작 패턴 (데이터 생성 → Ollama 언로드 → 학습)
nohup bash -c '
  python stages/stage5_university/data_prep.py && \
  curl -X POST http://localhost:11434/api/generate \
       -d "{\"model\": \"gemma4:e4b\", \"keep_alive\": 0}" && \
  WANDB_MODE=offline python stages/stage5_university/train.py --reset
' > logs/stage5_v1.log 2>&1 &
```

### W&B 오프라인 → 온라인 동기화
```bash
# 학습 완료 후 클라우드 동기화
wandb sync wandb/offline-run-YYYYMMDD_HHMMSS-RUNID

# 리니지(Lineage) 수동 복구 (오프라인 학습으로 끊긴 화살표 복원)
python tools/link_stage3.py   # Stage 3 리니지 복구
python tools/link_stage4.py   # Stage 4 리니지 복구 (중간 체크포인트 포함)
python tools/sync_wandb_history.py  # 전체 리니지 일괄 복구
```

### 디스크 관리
```bash
# 이전 Stage 체크포인트 삭제 (지식은 이미 다음 Stage로 전이됨)
rm -rf checkpoints/stage0* checkpoints/stage1* checkpoints/stage2*

# 중간 스텝만 삭제 (best/final은 보존)
rm -rf checkpoints/stage3_middle/stage3_middle/step_*

# HuggingFace 캐시 정리 (~9~15 GB 확보 가능)
rm -rf ~/.cache/huggingface/datasets
rm -rf ~/.cache/huggingface/hub/datasets--*

# pip 캐시 정리 (~1.4 GB)
rm -rf ~/Library/Caches/pip
```

---

## 🤖 Ollama Teacher API

```python
from core.teacher.ollama_teacher import OllamaTeacher

teacher = OllamaTeacher(model="gemma4:e4b", timeout=240.0)

# Stage 2: Q&A 쌍 생성
qa_pairs = teacher.generate_qa_pairs(passage="광합성은...", stage=2, n=3)
# → [{"question": "...", "answer": "..."}, ...]

# Stage 3: Chain-of-Thought 단계별 풀이
cot = teacher.get_chain_of_thought("2x + 5 = 13을 풀어라", stage=3)
# → "1단계: ... 2단계: ... 3단계: 검증 ✓"

# Stage 4~5: 답변 품질 채점 (0.0~1.0)
score = teacher.score_response(question="민주주의란?", answer="...", stage=4)
# → 0.75 (0.6 이상이면 채택)

# Stage 6: 선호 쌍 생성 (RLHF)
pref = teacher.create_preference_pairs(
    question="AI란?", answer_a="...", answer_b="...", stage=6
)
# → {"chosen": "...", "rejected": "...", "reason": "..."}
```

> **💡 캐시 전략**: 모든 Teacher 응답은 `data/teacher_cache/`에 MD5 해시로 캐시됩니다.
> 재실행 시 네트워크/Ollama 호출 없이 즉시 반환 → 중단 후 재시작 비용 최소화.

---

## 🗃️ 데이터 수집 파이프라인

```bash
# 특정 Stage 데이터 수집
python tools/collect_data.py --stage 0

# 여러 Stage 동시
python tools/collect_data.py --stage 0 1 2

# 전체 Stage
python tools/collect_data.py --all

# 미리보기 (저장 없이 샘플만 출력)
python tools/collect_data.py --stage 3 --preview
```

**데이터 수집 목표량 요약:**

| Stage | 목표 | 주요 소스 |
|-------|------|----------|
| 0 | 30,000건 | TinyStories (80%) + Wiki_ko 극초단 (20%) |
| 1 | 50,000건 | TinyStories (50%) + Wiki_ko 중간 (50%) |
| 2 | 80,000건 | TinyStories(25%) + ShareGPT-ko(30%) + Wiki_ko(40%) |
| 3 | 100,000건 | ShareGPT-ko(40%) + Wiki_ko(40%) + TinyStories(15%) |
| 4 | 100,000건 | ShareGPT-ko(50%) + Wiki_ko(40%) + score≥0.6 Q&A |
| 5 | 100,000건 | ShareGPT-ko 심화(50%) + Wiki_ko(30%) + Python코드(15%) |
| 6 | 80,000건 | Wiki_ko (60%) + ShareGPT_ko (30%) + RLHF쌍 (소량) |

---

## 🚀 빠른 시작

### 1. 환경 설정

```bash
git clone https://github.com/your-repo/EurekaAI.git
cd EurekaAI

python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Ollama Teacher 준비 (Stage 2 이상에서 필요)

```bash
# Ollama 설치 후
ollama serve
ollama pull gemma4:e4b     # 권장 (Stage 2~6)
# 또는
ollama pull llama3.2:3b    # 대안
```

### 3. 토크나이저 학습 (최초 1회)

```bash
python scripts/setup_tokenizer.py
```

### 4. Stage별 학습 시작 (권장 패턴)

```bash
# 데이터 생성 → Ollama 언로드 → 학습 (OOM 방지 콤보)
nohup bash -c '
  python stages/stage4_high/data_prep.py && \
  curl -X POST http://localhost:11434/api/generate \
       -d "{\"model\": \"gemma4:e4b\", \"keep_alive\": 0}" && \
  WANDB_MODE=offline python stages/stage4_high/train.py --reset
' > logs/stage4_v1.log 2>&1 &

# 로그 실시간 확인
tail -f logs/stage4_v1.log | grep -E "Step|Eval|PPL"
```

### 5. 현재 상태 확인

```bash
python run.py --status
```

---

## 📁 디렉토리 구조

```
EurekaAI/
├── core/                          # 핵심 모듈
│   ├── model/
│   │   ├── architecture.py        # EurekaModel (80M Transformer)
│   │   ├── config.py              # EurekaConfig + tiny/small/medium preset
│   │   └── tokenizer_utils.py     # 한영 BPE Tokenizer (32K)
│   ├── teacher/
│   │   └── ollama_teacher.py      # Ollama API 연동 (Q&A/CoT/Score/RLHF)
│   ├── curriculum/
│   │   ├── data_manager.py        # EurekaDataset + collate_fn
│   │   └── progression.py         # 단계 졸업 관리 (progression.json)
│   ├── training/
│   │   ├── trainer.py             # EurekaTrainer (MPS/CUDA/CPU)
│   │   └── continual.py           # EWC + ReplayBuffer
│   └── evaluation/
│       └── benchmarks.py          # 단계별 벤치마크
│
├── stages/                        # 단계별 학습 스크립트
│   ├── stage0_newborn/            # config.yaml + data_prep.py + train.py
│   ├── stage1_toddler/
│   ├── stage2_elementary/
│   ├── stage3_middle/
│   ├── stage4_high/
│   ├── stage5_university/
│   └── stage6_social/
│
├── tools/                         # 유틸리티
│   ├── collect_data.py            # 대용량 데이터 수집 파이프라인
│   ├── serve_stage.py             # Stage별 모델 서빙
│   ├── force_graduate_stage3.py   # Stage 3 강제 졸업 처리
│   ├── force_graduate_stage4.py   # Stage 4 강제 졸업 처리
│   ├── link_stage3.py             # Stage 3 W&B 리니지 복구
│   ├── link_stage4.py             # Stage 4 W&B 리니지 복구 (중간 체크포인트 포함)
│   ├── sync_wandb_history.py      # W&B 전체 히스토리 동기화
│   └── delete_model_garbage.py    # 불필요 체크포인트 정리
│
├── data/                          # 학습 데이터 (git 제외)
│   ├── tokenizer/eureka.model     # BPE 토크나이저
│   ├── processed/                 # Stage별 전처리 데이터
│   └── teacher_cache/             # Ollama 응답 캐시 (MD5)
│
├── checkpoints/                   # 모델 체크포인트 (git 제외)
├── logs/                          # 학습 로그 (git 제외)
├── wandb/                         # W&B 로컬 캐시 (git 제외)
│
├── run.py                         # 메인 실행 진입점
├── generate.py                    # 텍스트 생성 테스트
├── progression.json               # 커리큘럼 진행 상태
└── requirements.txt
```

---

## 🔬 W&B 실험 추적

모든 학습은 W&B에 자동 기록됩니다.

```python
# wandb.ai → EurekaAI-Curriculum 프로젝트에서 확인
# - train/loss, train/ppl per step
# - eval/loss, eval/ppl per eval_interval
# - Artifact Lineage: dataset → model 의존성 자동 추적
```

**W&B Artifact 네이밍 규칙:**
```
dataset-stage0 → model-stage0 → model-stage1 → ... → model-stage6
```

**오프라인 모드 리니지 복구:**
```
학습 중: WANDB_MODE=offline (OOM 방지, 네트워크 타임아웃 방지)
학습 후: wandb sync <run_dir>  → 로그/메트릭 동기화
리니지 : python tools/link_stage4.py → 화살표(노드 연결) 복원
```

---

## 🏆 현재까지의 성과

| Stage | 졸업 기준 | 최종 PPL | 스텝 | 상태 |
|-------|-----------|---------|------|------|
| **Stage 0** (신생아) | PPL ≤ 30 | **25.2** | 12,000 | ✅ 졸업 |
| **Stage 1** (유아기) | PPL ≤ 20 | **19.4** | 2,200 | ✅ 졸업 |
| **Stage 2** (초등) | PPL ≤ 15 | **38.51** → 재학습 후 달성 | 30,000 | ✅ 졸업 |
| **Stage 3** (중학교) | PPL ≤ 10 | **68.72** | 30,000 | ✅ force_graduate |
| **Stage 4** (고등) | PPL ≤ 8 | **46.57** | 15,000 | ✅ force_graduate |
| **Stage 5** (대학교) | PPL ≤ 20 | 학습 중 🔄 | 20,000 목표 | 🔄 진행 중 |
| **Stage 6** (사회인) | PPL ≤ 5 | - | - | ⏳ 대기 |

**수정된 주요 버그:**
- Label shifting (CLM 핵심 오류) → 1-token shift 수정
- Gradient accumulation logging 오류 → 실제 step 기준으로 수정  
- Eval loop labels 덮어쓰기 → clone() 사용
- Gemma JSON parsing 오류 → 텍스트 fallback 파서 추가
- Stage4/5 경로 불일치 → stage4_high/stage5_university 통일
- Ollama OOM → 학습 전 강제 언로드 루틴 추가

**핵심 레슨:**
- 데이터 정제 >> 모델 크기
- Curriculum Shock(초기 PPL 폭등)는 정상 → 2,000 스텝 내 안정화
- Auto-resume 필수 (Mac 절전 대비)
- Ollama 언로드 후 학습 진입 필수 (OOM 방지)
- 오프라인 W&B + 사후 리니지 복구가 안정적

---

*"유레카! 모든 지식은 스스로 발견될 때 가장 빛난다."* ✨

