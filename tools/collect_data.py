#!/usr/bin/env python3
"""
EurekaAI — 대용량 데이터 수집 파이프라인 (120M 스케일업용)
=============================================================
Stage별 목표 샘플 수를 달성하기 위해 HuggingFace 데이터셋에서
대용량 말뭉치를 수집, 정제, 저장합니다.

목표:
  Stage 0:  30,000건 (TinyStories 동화)
  Stage 1:  50,000건 (동화 + 짧은 대화)
  Stage 2:  80,000건 (쉬운 위키 단락)
  Stage 3: 100,000건 (위키 단락 + 논리글)
  Stage 4: 100,000건 (위키 심화 + Q&A)
  Stage 5: 100,000건 (학술 위키 + Q&A)
  Stage 6:  80,000건 (대화 + 코드 + 위키)

사용법:
    python tools/collect_data.py --stage 0        # 특정 Stage만
    python tools/collect_data.py --all            # 전체 Stage
    python tools/collect_data.py --stage 0 --preview  # 샘플 미리보기만
"""

import sys, json, random, logging, argparse, re
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

Path("logs").mkdir(exist_ok=True)
_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
logging.basicConfig(
    level=logging.INFO,
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(f"logs/collect_data_{_ts}.log", encoding="utf-8"),
    ],
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ── 공통 유틸 ─────────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """불필요한 위키 마크업, 특수문자 정리."""
    text = re.sub(r"\[\[([^|\]]+\|)?([^\]]+)\]\]", r"\2", text)  # [[link|text]] → text
    text = re.sub(r"\{\{[^}]+\}\}", "", text)                      # {{template}} 제거
    text = re.sub(r"={2,}[^=]+=*", "", text)                       # == 제목 == 제거
    text = re.sub(r"<[^>]+>", "", text)                            # HTML 태그 제거
    text = re.sub(r"https?://\S+", "", text)                       # URL 제거
    text = re.sub(r"\s+", " ", text).strip()
    return text


def filter_para(text: str, min_len: int = 200, max_len: int = 1000) -> bool:
    """단락 품질 필터."""
    if len(text) < min_len or len(text) > max_len:
        return False
    # 한국어 또는 영어 비율 확인 (특수문자 도배 방지)
    alpha_ratio = sum(1 for c in text if c.isalpha()) / len(text)
    if alpha_ratio < 0.5:
        return False
    return True


def save_jsonl(samples: list[dict], path: Path, eval_ratio: float = 0.1):
    """Train/Eval 분리 저장."""
    path.parent.mkdir(parents=True, exist_ok=True)
    random.shuffle(samples)
    n_eval = min(2000, int(len(samples) * eval_ratio))

    eval_path = path.parent / path.name.replace("train_", "eval_")
    with open(path, "w", encoding="utf-8") as f:
        for s in samples[n_eval:]:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    with open(eval_path, "w", encoding="utf-8") as f:
        for s in samples[:n_eval]:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    logger.info(f"💾 저장: Train {len(samples)-n_eval:,}건 → {path}")
    logger.info(f"💾 저장: Eval  {n_eval:,}건 → {eval_path}")
    return len(samples) - n_eval, n_eval


def load_wikipedia_ko(target: int, min_len: int = 200, max_len: int = 800,
                      wiki_split: str = "train[:3000]") -> list[dict]:
    """한국어 위키백과에서 단락 수집."""
    from datasets import load_dataset
    logger.info(f"📥 Wikipedia (ko) 로딩... (split={wiki_split})")
    ds = load_dataset("wikimedia/wikipedia", "20231101.ko", split=wiki_split)

    samples = []
    for item in ds:
        if len(samples) >= target:
            break
        for para in item.get("text", "").split("\n"):
            para = clean_text(para.strip())
            if filter_para(para, min_len, max_len):
                samples.append({"text": para, "source": "wiki_ko"})
            if len(samples) >= target:
                break

    random.shuffle(samples)
    logger.info(f"  Wikipedia (ko): {len(samples):,}건 수집")
    return samples


# ── Stage 0: 신생아 (TinyStories) ─────────────────────────────────────────────

def collect_stage0(target: int = 30_000, preview: bool = False) -> list[dict]:
    """TinyStories + 한국어 짧은 동화."""
    from datasets import load_dataset
    samples = []

    # TinyStories (영어 동화, 매우 단순한 문장)
    logger.info("📥 TinyStories 로딩...")
    try:
        ts_target = int(target * 0.6)
        # 필요한 양의 3배 슬라이스로 넉넉하게 로드 후 필터링
        ds = load_dataset("roneneldan/TinyStories", split=f"train[:{ts_target * 3}]")
        for item in ds:
            text = item.get("text", "").strip()
            if 100 <= len(text) <= 600:
                samples.append({"text": text[:500], "source": "tinystories", "stage": 0})
            if len(samples) >= ts_target:
                break
        logger.info(f"  TinyStories: {len(samples):,}건")
    except Exception as e:
        logger.warning(f"  TinyStories 실패: {e}")

    # 한국어 위키 짧은 단락 (아동 수준, 100~300자)
    try:
        ko_samples = load_wikipedia_ko(
            target=int(target * 0.4),
            min_len=100, max_len=300,
            wiki_split="train[:500]"
        )
        samples.extend(ko_samples)
    except Exception as e:
        logger.warning(f"  Wiki(ko) 실패: {e}")

    if preview:
        for s in samples[:3]:
            print(f"  [{s['source']}] {s['text'][:100]}")
    return samples[:target]


# ── Stage 1: 유아 ──────────────────────────────────────────────────────────────

def collect_stage1(target: int = 50_000, preview: bool = False) -> list[dict]:
    """동화 + 짧은 대화 혼합."""
    from datasets import load_dataset
    samples = []

    # TinyStories (조금 더 긴 것)
    try:
        ts_target = int(target * 0.5)
        ds = load_dataset("roneneldan/TinyStories", split=f"train[:{ts_target * 3}]")
        for item in ds:
            text = item.get("text", "").strip()
            if 150 <= len(text) <= 800:
                samples.append({"text": text[:700], "source": "tinystories", "stage": 1})
            if len(samples) >= ts_target:
                break
        logger.info(f"  TinyStories (Stage1): {len(samples):,}건")
    except Exception as e:
        logger.warning(f"  TinyStories 실패: {e}")

    # 한국어 위키 (150~400자)
    try:
        ko = load_wikipedia_ko(int(target * 0.5), min_len=150, max_len=400,
                               wiki_split="train[:800]")
        samples.extend(ko)
    except Exception as e:
        logger.warning(f"  Wiki(ko) 실패: {e}")

    if preview:
        for s in samples[:3]:
            print(f"  [{s['source']}] {s['text'][:100]}")
    return samples[:target]


# ── Stage 2: 초등 ──────────────────────────────────────────────────────────────

def collect_stage2(target: int = 80_000, preview: bool = False) -> list[dict]:
    """쉬운 위키 단락 (200~500자)."""
    samples = []
    try:
        samples = load_wikipedia_ko(target, min_len=200, max_len=500,
                                    wiki_split="train[:5000]")
    except Exception as e:
        logger.warning(f"  Stage2 데이터 수집 실패: {e}")
    for s in samples:
        s["stage"] = 2
    if preview:
        for s in samples[:3]:
            print(f"  [{s['source']}] {s['text'][:100]}")
    return samples[:target]


# ── Stage 3: 중등 ──────────────────────────────────────────────────────────────

def collect_stage3(target: int = 100_000, preview: bool = False) -> list[dict]:
    """위키 단락 (250~600자) — 논리적 내용 중심."""
    samples = []
    try:
        samples = load_wikipedia_ko(target, min_len=250, max_len=600,
                                    wiki_split="train[:8000]")
    except Exception as e:
        logger.warning(f"  Stage3 데이터 수집 실패: {e}")
    for s in samples:
        s["stage"] = 3
    if preview:
        for s in samples[:3]:
            print(f"  [{s['source']}] {s['text'][:100]}")
    return samples[:target]


# ── Stage 4: 고등 ──────────────────────────────────────────────────────────────

def collect_stage4(target: int = 100_000, preview: bool = False) -> list[dict]:
    """위키 심화 단락 (300~800자) + Teacher Q&A."""
    samples = []
    # 위키 심화 단락
    try:
        wiki = load_wikipedia_ko(int(target * 0.85), min_len=300, max_len=800,
                                 wiki_split="train[:10000]")
        samples.extend(wiki)
    except Exception as e:
        logger.warning(f"  Stage4 위키 실패: {e}")

    # 기존 stage4 데이터 재활용 (있는 경우)
    existing = Path("data/processed/stage4_high/train_stage4_high.jsonl")
    if not existing.exists():
        existing = Path("data/processed/stage4/train_stage4.jsonl")
    if existing.exists():
        with open(existing) as f:
            existing_data = [json.loads(l) for l in f if l.strip()]
        # 300자+ 필터
        existing_data = [d for d in existing_data if len(d.get("text","")) >= 300]
        samples.extend(existing_data[:int(target * 0.15)])
        logger.info(f"  Stage4 기존 데이터 재활용: {len(existing_data):,}건")

    for s in samples:
        s["stage"] = 4
    if preview:
        for s in samples[:3]:
            print(f"  [{s['source']}] {s['text'][:100]}")
    return samples[:target]


# ── Stage 5: 대학교 ────────────────────────────────────────────────────────────

def collect_stage5(target: int = 100_000, preview: bool = False) -> list[dict]:
    """학술 위키 단락 (350~1000자) + 기존 Q&A 재활용."""
    samples = []
    # 학술적 긴 단락
    try:
        wiki = load_wikipedia_ko(int(target * 0.8), min_len=350, max_len=1000,
                                 wiki_split="train[:15000]")
        samples.extend(wiki)
    except Exception as e:
        logger.warning(f"  Stage5 위키 실패: {e}")

    # 기존 Stage5 고품질 Q&A 재활용
    for fname in ["data/processed/stage5/train_stage5_filtered.jsonl",
                  "data/processed/stage5/train_stage5.jsonl"]:
        existing = Path(fname)
        if existing.exists():
            with open(existing) as f:
                qa_data = [json.loads(l) for l in f if l.strip()]
            qa_data = [d for d in qa_data if d.get("text","").startswith("Q:") and len(d.get("text","")) >= 300]
            samples.extend(qa_data)
            logger.info(f"  Stage5 Q&A 재활용: {len(qa_data):,}건")
            break

    for s in samples:
        s["stage"] = 5
    if preview:
        for s in samples[:3]:
            print(f"  [{s['source']}] {s['text'][:100]}")
    return samples[:target]


# ── Stage 6: 사회인 ────────────────────────────────────────────────────────────

def collect_stage6(target: int = 80_000, preview: bool = False) -> list[dict]:
    """대화 + 코드 + 위키 혼합."""
    from datasets import load_dataset
    samples = []

    # 위키 (300~800자)
    try:
        wiki = load_wikipedia_ko(int(target * 0.6), min_len=300, max_len=800,
                                 wiki_split="train[:8000]")
        samples.extend(wiki)
    except Exception as e:
        logger.warning(f"  Stage6 위키 실패: {e}")

    # ShareGPT 한국어 대화 시도
    try:
        sg_target = int(target * 0.3)
        ds = load_dataset("junelee/sharegpt_deepl_ko", split=f"train[:{sg_target * 3}]")
        count = 0
        for item in ds:
            convs = item.get("conversations", [])
            if len(convs) >= 2:
                text = f"User: {convs[0].get('value','')[:200]}\nAssistant: {convs[1].get('value','')[:400]}"
                if len(text) >= 200:
                    samples.append({"text": text, "source": "sharegpt_ko", "stage": 6})
                    count += 1
            if count >= sg_target:
                break
        logger.info(f"  ShareGPT (ko): {count:,}건")
    except Exception as e:
        logger.warning(f"  ShareGPT 실패: {e} (위키로 대체)")

    # 기존 Stage6 데이터 보강
    existing = Path("data/processed/stage6/train_stage6.jsonl")
    if existing.exists():
        with open(existing) as f:
            ex = [json.loads(l) for l in f if l.strip()]
        samples.extend(ex)
        logger.info(f"  Stage6 기존 데이터 재활용: {len(ex):,}건")

    for s in samples:
        s["stage"] = 6
    if preview:
        for s in samples[:3]:
            print(f"  [{s['source']}] {s['text'][:100]}")
    return samples[:target]


# ── 공통 처리 & 저장 ──────────────────────────────────────────────────────────

STAGE_COLLECTORS = {
    0: (collect_stage0, 30_000,  "stage0_newborn",    "train_stage0_newborn.jsonl"),
    1: (collect_stage1, 50_000,  "stage1_toddler",    "train_stage1_toddler.jsonl"),
    2: (collect_stage2, 80_000,  "stage2_elementary", "train_stage2_elementary.jsonl"),
    3: (collect_stage3, 100_000, "stage3_middle",     "train_stage3_middle.jsonl"),
    4: (collect_stage4, 100_000, "stage4_high",       "train_stage4_high.jsonl"),
    5: (collect_stage5, 100_000, "stage5_university", "train_stage5_university.jsonl"),
    6: (collect_stage6, 80_000,  "stage6_social",     "train_stage6_social.jsonl"),
}


def run_stage(stage: int, preview: bool = False):
    collector, target, dir_name, filename = STAGE_COLLECTORS[stage]
    logger.info("=" * 60)
    logger.info(f"  Stage {stage}: {dir_name}  (목표: {target:,}건)")
    logger.info("=" * 60)

    samples = collector(target=target, preview=preview)

    if preview:
        logger.info(f"  [미리보기] {len(samples):,}건 수집됨 (저장 안 함)")
        return

    # 중복 제거 (첫 80자 기준)
    seen = set()
    deduped = []
    for s in samples:
        key = s["text"][:80]
        if key not in seen:
            seen.add(key)
            deduped.append(s)

    from collections import Counter
    logger.info(f"\n📊 데이터 구성 ({len(deduped):,}건):")
    for src, cnt in Counter(s["source"] for s in deduped).most_common():
        logger.info(f"   {src:25s}: {cnt:6,} ({cnt/len(deduped)*100:.1f}%)")

    import statistics
    lengths = [len(s["text"]) for s in deduped]
    logger.info(f"   평균 길이: {statistics.mean(lengths):.0f}자  |  중앙값: {statistics.median(lengths):.0f}자")

    out_path = Path(f"data/processed/{dir_name}/{filename}")
    n_train, n_eval = save_jsonl(deduped, out_path)
    logger.info(f"\n✅ Stage {stage} 완료: Train {n_train:,} / Eval {n_eval:,}")
    return n_train, n_eval


def main():
    parser = argparse.ArgumentParser(
        description="EurekaAI 대용량 데이터 수집 파이프라인 (120M 스케일업)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  python tools/collect_data.py --all              # 전체 Stage 수집
  python tools/collect_data.py --stage 0          # Stage 0만
  python tools/collect_data.py --stage 0 1 2      # 여러 Stage
  python tools/collect_data.py --stage 5 --preview # 미리보기
        """
    )
    parser.add_argument("--stage",   type=int, nargs="+", choices=range(7),
                        help="수집할 Stage 번호 (복수 지정 가능)")
    parser.add_argument("--all",     action="store_true", help="전체 Stage 수집")
    parser.add_argument("--preview", action="store_true", help="샘플만 출력, 저장 안 함")
    args = parser.parse_args()

    if args.all:
        stages = list(range(7))
    elif args.stage:
        stages = args.stage
    else:
        parser.print_help()
        return

    logger.info(f"🚀 EurekaAI 데이터 수집 시작: Stage {stages}")
    logger.info(f"   로그: logs/collect_data_{_ts}.log\n")

    results = {}
    for stage in stages:
        try:
            result = run_stage(stage, preview=args.preview)
            if result:
                results[stage] = result
        except Exception as e:
            logger.error(f"Stage {stage} 실패: {e}", exc_info=True)

    if results:
        logger.info("\n" + "=" * 60)
        logger.info("  최종 수집 결과")
        logger.info("=" * 60)
        total_train = sum(r[0] for r in results.values())
        total_eval  = sum(r[1] for r in results.values())
        for s, (tr, ev) in results.items():
            logger.info(f"  Stage {s}: Train {tr:,} / Eval {ev:,}")
        logger.info(f"  합계:    Train {total_train:,} / Eval {total_eval:,}")


if __name__ == "__main__":
    main()
