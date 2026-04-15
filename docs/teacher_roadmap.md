# Teacher 모델 (Ollama) 활용 로드맵

> **핵심 원칙**: Teacher는 Stage 2 (초등학교) 이상부터 본격 활용한다.
> Stage 0~1은 단순 텍스트 패턴 학습이므로 Teacher 불필요.

---

## Stage별 Teacher 활용 계획

| Stage | 이름 | Teacher 역할 | 사용 메서드 |
|-------|------|-------------|------------|
| 0 | 🍼 신생아 | ❌ 미사용 (단순 CLM만) | - |
| 1 | 🧸 유아기 | ❌ 미사용 (단순 CLM만) | - |
| **2** | **📚 초등학교** | **✅ Q&A 데이터 생성** | `generate_qa_pairs()` |
| **3** | **🔢 중학교** | **✅ Chain-of-Thought 증류** | `get_chain_of_thought()` |
| **4** | **📐 고등학교** | **✅ CoT + Q&A + 채점** | `score_response()` |
| **5** | **🎓 대학교** | **✅ RLHF 보상 신호** | `score_response()` |
| **6** | **🌐 사회인** | **✅ 선호 데이터 + RLHF** | `create_preference_pairs()` |

---

## Stage 2 data_prep.py 구현 시 해야 할 것

```python
# Stage 2 data_prep.py 에서 Teacher 활용 예시
from core.teacher.ollama_teacher import OllamaTeacher

teacher = OllamaTeacher(model="gemma4:e4b")

# 1. 지문에서 Q&A 생성
qa_pairs = teacher.generate_qa_pairs(passage=text, stage=2, n=5)

# 2. 합성 초등 수준 텍스트 생성
synthetic = teacher.generate_synthetic_stage_data(stage=2, n_samples=1000)
```

## Stage 3 data_prep.py 구현 시 해야 할 것

```python
# Chain-of-Thought 데이터 생성 (수학/논리 문제)
cot = teacher.get_chain_of_thought(problem="2+3은 얼마인가요?", stage=3)
```

## Stage 5~6 training 시 해야 할 것

```python
# RLHF: EurekaAI 답변 채점 → 보상 신호
score = teacher.score_response(question=q, answer=student_ans, stage=5)

# RLHF: 선호 데이터 생성
pref = teacher.create_preference_pairs(question=q, answer_a=a1, answer_b=a2, stage=6)
```

---

## 주의사항

- Teacher는 **로컬 Ollama** 필요: `ollama serve` + `ollama pull gemma4:e4b`
- 응답 캐싱 활성화됨 (`data/teacher_cache/`) → 반복 호출 비용 절약
- Ollama 미사용 시 자동 스킵 (`is_available()` 체크)
