"""
EurekaAI — Stage 3 (Middle School) Data Preparation
중학교 단계: 논리적 추론 + Chain-of-Thought 학습
  - Teacher.get_chain_of_thought()로 단계별 풀이 데이터 생성 ← 핵심!
  - 수학/과학 기초 문제
  - 이전 단계 리플레이 10%
"""

import sys
import json
import random
import logging
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from core.teacher.ollama_teacher import OllamaTeacher

# ── 로그 설정 ─────────────────────────────────────────────────────────────────
Path("logs").mkdir(exist_ok=True)
_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
_log_file = f"logs/stage3_data_prep_{_ts}.log"
_fmt = "%(asctime)s [%(levelname)s] %(message)s"
logging.basicConfig(level=logging.INFO, handlers=[
    logging.StreamHandler(sys.stdout),
    logging.FileHandler(_log_file, encoding="utf-8"),
], format=_fmt)
logger = logging.getLogger(__name__)
logger.info(f"📄 로그: {_log_file}")

STAGE_DIR  = Path("data/processed/stage3")
TRAIN_FILE = STAGE_DIR / "train_stage3.jsonl"
EVAL_FILE  = STAGE_DIR / "eval_stage3.jsonl"
TARGET_EVAL = 2000

# 이전 단계 데이터 경로
PREV_STAGES = [
    Path("data/processed/stage0/train_stage0.jsonl"),
    Path("data/processed/stage1/train_stage1.jsonl"),
    Path("data/processed/stage2/train_stage2.jsonl"),
]

# ── CoT 문제 목록 ─────────────────────────────────────────────────────────────
MATH_PROBLEMS_KO = [
    "2x + 5 = 13일 때 x를 구하세요.",
    "삼각형의 넓이를 구하는 공식은 무엇인가요? 밑변 6, 높이 4의 넓이는?",
    "1부터 100까지의 합을 구하세요.",
    "원의 둘레는 어떻게 구하나요? 반지름이 7cm인 원의 둘레는?",
    "소수란 무엇인가요? 1~20 사이의 소수를 모두 나열하세요.",
    "3의 배수이면서 5의 배수인 수 중 100 이하의 수를 구하세요.",
    "두 자리 수 중 각 자리 숫자의 합이 10인 수를 모두 구하세요.",
    "비율로 나타내기: 반 30명 중 여학생이 18명이면 여학생의 비율은?",
    "평균 구하기: 70, 80, 90, 85, 75점의 평균은?",
    "비례식 풀기: 3:4 = x:20에서 x를 구하세요.",
]

SCIENCE_PROBLEMS_KO = [
    "광합성 과정을 단계별로 설명해주세요.",
    "뉴턴의 운동 법칙 3가지를 설명해주세요.",
    "물의 상태 변화(고체, 액체, 기체)를 설명해주세요.",
    "산성, 염기성, 중성의 차이는 무엇인가요?",
    "지구의 자전과 공전의 차이를 설명해주세요.",
    "먹이사슬이란 무엇인가요? 예를 들어 설명해주세요.",
    "원소와 화합물의 차이점은 무엇인가요?",
    "전기 회로에서 직렬연결과 병렬연결의 차이는?",
]

LOGIC_PROBLEMS = [
    "All cats are animals. All animals breathe. Do cats breathe? Explain your reasoning.",
    "If today is Monday, what day will it be 100 days from now?",
    "A farmer has 17 sheep. All but 9 run away. How many are left?",
    "What comes next in this pattern: 2, 4, 8, 16, 32, ?",
    "If you have a 3-liter jug and a 5-liter jug, how can you measure exactly 4 liters?",
]


def format_cot(problem: str, solution: str) -> str:
    return f"문제: {problem}\n\n풀이:\n{solution}"


def generate_cot_data(max_workers: int = 2) -> list[dict]:
    """Teacher.get_chain_of_thought()로 단계별 풀이 생성 — Stage 3 핵심!"""
    teacher = OllamaTeacher()
    if not teacher.is_available():
        logger.warning("⚠️  Ollama 미사용 — CoT 데이터 스킵")
        return []

    all_problems = [
        (p, 3) for p in MATH_PROBLEMS_KO
    ] + [
        (p, 3) for p in SCIENCE_PROBLEMS_KO
    ] + [
        (p, 3) for p in LOGIC_PROBLEMS
    ]
    # 반복 확장
    extended = all_problems * 8  # 각 문제 8번 반복 (캐시 재활용)
    random.shuffle(extended)

    logger.info(f"🤖 CoT 데이터 생성 ({len(extended)}개 문제, 병렬 {max_workers}개)...")

    def process_one(args):
        problem, stage = args
        try:
            solution = teacher.get_chain_of_thought(problem=problem, stage=stage)
            if solution and len(solution) > 20:
                return {"text": format_cot(problem, solution), "source": "teacher_cot", "stage": 3}
        except Exception as e:
            logger.debug(f"CoT 실패: {e}")
        return None

    samples = []
    completed = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_one, p): p for p in extended}
        for future in as_completed(futures):
            completed += 1
            result = future.result()
            if result:
                samples.append(result)
            if completed % 20 == 0:
                logger.info(f"  완료: {completed}/{len(extended)} | CoT {len(samples)}건")

    logger.info(f"✅ Teacher CoT: {len(samples):,}건")
    return samples


def load_wiki_paragraphs(max_samples: int = 12000) -> list[dict]:
    """한국어 위키 — 중학교 수준 단락 (더 길고 논리적)"""
    samples = []
    try:
        from datasets import load_dataset
        logger.info(f"📥 Korean Wikipedia ({max_samples:,}건)...")
        ko_ds = load_dataset("wikimedia/wikipedia", "20231101.ko", split="train[:600]")
        for item in ko_ds:
            if len(samples) >= max_samples:
                break
            text = item.get("text", "").strip()
            # 단락 단위 (문장보다 긴 내용)
            for para in text.split("\n"):
                para = para.strip()
                if 80 <= len(para) <= 400:
                    samples.append({"text": para, "source": "wiki_ko", "stage": 3})
                if len(samples) >= max_samples:
                    break
        random.shuffle(samples)
        logger.info(f"✅ Wikipedia 단락: {len(samples):,}건")
    except Exception as e:
        logger.warning(f"⚠️  Wikipedia 실패: {e}")
    return samples


def load_replay(max_per_stage: int = 3000) -> list[dict]:
    """이전 단계 데이터 리플레이 — 망각 방지"""
    samples = []
    for path in PREV_STAGES:
        if not path.exists():
            continue
        stage_name = path.parent.name
        lines = path.read_text().strip().split("\n")
        random.shuffle(lines)
        n = min(max_per_stage, int(len(lines) * 0.10))
        for line in lines[:n]:
            d = json.loads(line)
            d["source"] = f"{stage_name}_replay"
            d["stage"] = 3
            samples.append(d)
    logger.info(f"✅ 이전 단계 리플레이: {len(samples):,}건")
    return samples


def generate_synthetic(n: int = 300) -> list[dict]:
    teacher = OllamaTeacher()
    if not teacher.is_available():
        return []
    raw = teacher.generate_synthetic_stage_data(stage=3, n_samples=n, language="mixed")
    return [{**s, "source": "ollama_synthetic", "stage": 3} for s in raw if s.get("text", "").strip()]


def save_jsonl(samples: list[dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    logger.info(f"💾 저장: {len(samples):,}건 → {path}")


def main():
    logger.info("=" * 60)
    logger.info("   EurekaAI — Stage 3: Middle School Data Preparation")
    logger.info("=" * 60)

    all_samples = []
    all_samples += generate_cot_data(max_workers=2)       # CoT 핵심 ★
    all_samples += load_wiki_paragraphs(max_samples=12000)
    all_samples += load_replay(max_per_stage=3000)
    all_samples += generate_synthetic(n=300)

    # 중복 제거 + 셔플
    seen = set()
    deduped = [s for s in all_samples if (k := s["text"][:80]) not in seen and not seen.add(k)]
    random.shuffle(deduped)

    from collections import Counter
    total = len(deduped)
    logger.info(f"\n📊 데이터 구성 ({total:,}건):")
    for src, cnt in Counter(s["source"] for s in deduped).most_common():
        logger.info(f"   {src:25s}: {cnt:6,} ({cnt/total*100:.1f}%)")

    n_eval = min(TARGET_EVAL, int(total * 0.1))
    random.shuffle(deduped)
    save_jsonl(deduped[n_eval:], TRAIN_FILE)
    save_jsonl(deduped[:n_eval], EVAL_FILE)
    logger.info(f"\n✅ Stage 3 데이터 완료: Train {total-n_eval:,} / Eval {n_eval:,}")


if __name__ == "__main__":
    main()
