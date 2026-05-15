"""
EurekaAI — Stage 6 (Social) Data Preparation
사회인 단계: 자연스러운 대화, 코드, 복잡 문제 + RLHF 선호 데이터
  - Teacher.create_preference_pairs()로 선호 데이터 생성 ← Stage 6 핵심!
  - 대화형 데이터, 코드 설명, 뉴스 스타일
"""
import sys, json, random, logging
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from core.teacher.ollama_teacher import OllamaTeacher

Path("logs").mkdir(exist_ok=True)
_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
_log_file = f"logs/stage6_data_prep_{_ts}.log"
logging.basicConfig(level=logging.INFO, handlers=[
    logging.StreamHandler(sys.stdout),
    logging.FileHandler(_log_file, encoding="utf-8"),
], format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)
logger.info(f"📄 로그: {_log_file}")

STAGE_DIR  = Path("data/processed/stage6_social")
TRAIN_FILE = STAGE_DIR / "train_stage6_social.jsonl"
EVAL_FILE  = STAGE_DIR / "eval_stage6_social.jsonl"
TARGET_EVAL = 2000
# 실제 저장 경로에 맞게 수정 (stage3_middle, stage4_high, stage5_university)
PREV_STAGES = [
    Path("data/processed/stage3_middle/train_stage3_middle.jsonl"),
    Path("data/processed/stage4_high/train_stage4_high.jsonl"),
    Path("data/processed/stage5_university/train_stage5_university.jsonl"),
]

# 선호 데이터 생성용 질문 목록 (10 → 40개로 확장)
PREFERENCE_QUESTIONS = [
    # 🇰🇷 한국어 — AI/기술
    "최근 인공지능 기술의 발전이 사회에 미치는 영향은 무엇인가요?",
    "머신러닝과 딥러닝의 차이점을 설명해주세요.",
    "ChatGPT와 같은 대화형 AI를 개발하려면 어떤 기술이 필요한가요?",
    "빅데이터 분석이 기업 의사결정에 어떤 도움을 주나요?",
    "자율주행 자동차 기술의 현재 한계와 미래 전망을 설명해주세요.",
    # 🇰🇷 한국어 — 사회/윤리
    "기후 변화 문제를 해결하기 위한 현실적인 방안을 제시해주세요.",
    "좋은 리더십이란 무엇인지 설명해주세요.",
    "소셜 미디어가 청소년에게 미치는 긍정적, 부정적 영향을 분석해주세요.",
    "원격 근무의 장단점과 미래 직장 문화에 대해 설명해주세요.",
    "인구 고령화 문제를 해결하기 위한 정책 방안을 제안해주세요.",
    # 🇰🇷 한국어 — 코딩/개발
    "파이썬으로 피보나치 수열을 구현하는 방법을 설명해주세요.",
    "REST API와 GraphQL의 차이점은 무엇인가요?",
    "도커(Docker)와 쿠버네티스(Kubernetes)의 역할을 쉽게 설명해주세요.",
    "좋은 코드 리뷰를 하는 방법은 무엇인가요?",
    "데이터베이스 인덱스가 무엇이고 왜 중요한지 설명해주세요.",
    # 🇰🇷 한국어 — 경제/비즈니스
    "스타트업이 성공하기 위해 가장 중요한 요소는 무엇인가요?",
    "블록체인 기술이 금융 산업을 어떻게 변화시키고 있나요?",
    "지속 가능한 경영이란 무엇이며 왜 중요한가요?",
    "글로벌 공급망 위기의 원인과 해결책을 설명해주세요.",
    "ESG 경영이 기업 가치에 미치는 영향을 분석해주세요.",
    # 🌐 English — AI/Technology
    "What are the ethical implications of artificial intelligence?",
    "How do neural networks learn from data?",
    "What makes a good software architecture?",
    "Explain the concept of blockchain technology in simple terms.",
    "How can we make AI systems more interpretable and explainable?",
    # 🌐 English — Society
    "How can governments balance economic growth with environmental protection?",
    "What is the impact of social media on democracy and public discourse?",
    "How should we regulate large technology companies?",
    "What are the most pressing challenges facing global healthcare systems?",
    "How can education systems better prepare students for the future of work?",
    # 🌐 English — Coding
    "What are the best practices for writing clean, maintainable Python code?",
    "How do you design a scalable microservices architecture?",
    "Explain the difference between synchronous and asynchronous programming.",
    "What is the CAP theorem and why does it matter for distributed systems?",
    "How do you approach debugging a complex production issue?",
    # 🌐 English — Business
    "What factors determine the success of a technology startup?",
    "How should companies approach digital transformation?",
    "What is the role of data privacy in building customer trust?",
    "How can organizations foster a culture of innovation?",
    "What are the key principles of effective project management?",
]

CONVERSATION_TEMPLATES = [
    # 일상 대화
    ("안녕하세요! 오늘 날씨가 어때요?", "안녕하세요! 오늘은 맑고 따뜻한 날씨네요. 산책하기 딱 좋은 날이에요."),
    ("오늘 점심 뭐 먹을까요?", "한국 음식이 당기시면 김치찌개나 된장찌개 어떠세요? 가볍게 드시고 싶으면 샐러드나 샌드위치도 좋을 것 같아요."),
    ("요즘 많이 힘드네요.", "많이 지치셨군요. 힘든 일이 있으실 때는 잠시 쉬어가는 것도 중요해요. 어떤 부분이 특히 힘드셨나요?"),
    ("추천 도서가 있나요?", "관심 분야가 어디세요? 자기계발이라면 '아주 작은 습관의 힘', 소설이라면 한강 작가의 '채식주의자'를 추천드려요."),
    ("운동을 시작하려는데 어떻게 해야 할까요?", "처음엔 부담 없는 걷기나 가벼운 스트레칭부터 시작하세요. 일주일에 3회, 30분씩 꾸준히 하는 것이 중요해요."),
    # Python/개발
    ("파이썬에서 리스트와 튜플의 차이가 뭔가요?", "리스트는 변경 가능(mutable)하고 튜플은 변경 불가(immutable)합니다. 리스트는 [], 튜플은 ()를 사용해요. 딕셔너리 키로는 튜플만 사용 가능해요."),
    ("머신러닝 공부를 시작하려면 어떻게 해야 하나요?", "파이썬 기초를 먼저 익히고, numpy/pandas를 배운 후 scikit-learn으로 시작하세요. 캐글 대회도 큰 도움이 됩니다."),
    ("깃(Git)에서 브랜치를 어떻게 사용하나요?", "git branch feature/새기능으로 브랜치를 만들고, git checkout으로 전환해요. 작업 완료 후 git merge로 합치면 됩니다. PR(Pull Request)을 통해 코드 리뷰를 받는 것을 추천해요."),
    ("API 설계 시 가장 중요한 것은 무엇인가요?", "일관성, 명확성, 버전 관리가 핵심이에요. RESTful 원칙을 따르고, 에러 메시지를 명확하게 제공하며, 문서화를 철저히 해야 합니다."),
    ("SQL과 NoSQL의 차이를 설명해주세요.", "SQL은 정형화된 스키마의 관계형 DB(MySQL, PostgreSQL), NoSQL은 유연한 구조의 비관계형 DB(MongoDB, Redis)입니다. 데이터 구조가 명확하면 SQL, 유연성이 필요하면 NoSQL이 적합해요."),
    # AI/ML
    ("오버피팅이란 무엇인가요?", "모델이 훈련 데이터에는 잘 맞지만 새로운 데이터에는 성능이 떨어지는 현상이에요. Dropout, 정규화(L1/L2), 데이터 증강으로 방지할 수 있어요."),
    ("트랜스포머 모델이 뭔가요?", "어텐션 메커니즘을 핵심으로 하는 신경망 구조예요. BERT, GPT 등 최신 LLM의 기반이 되는 아키텍처로, 시퀀스 데이터 처리에 탁월합니다."),
    ("데이터 전처리 왜 중요한가요?", "'Garbage in, Garbage out'이라는 말처럼 데이터 품질이 모델 성능을 결정해요. 결측값 처리, 정규화, 이상치 제거 등이 핵심 전처리 단계입니다."),
    # English conversation
    ("What is the difference between supervised and unsupervised learning?", "Supervised learning uses labeled data where the model learns input-output mappings. Unsupervised learning finds patterns in unlabeled data through clustering or dimensionality reduction."),
    ("How do I optimize a Python function for performance?", "Use profiling tools like cProfile first, then consider NumPy for numerical operations, list comprehensions over loops, and functools.lru_cache for memoization."),
    ("Can you explain what REST API means?", "REST (Representational State Transfer) is an architectural style for APIs. It uses HTTP methods (GET, POST, PUT, DELETE), is stateless, and returns data in formats like JSON or XML."),
    ("What is the difference between AI, ML, and Deep Learning?", "AI is the broad concept of machines simulating intelligence. ML is a subset where machines learn from data. Deep Learning is a subset of ML using neural networks with many layers."),
    ("How should I structure a machine learning project?", "Follow the CRISP-DM framework: Business Understanding → Data Understanding → Data Preparation → Modeling → Evaluation → Deployment. Document each step and version your data and models."),
    ("What is Docker and why should I use it?", "Docker packages applications with their dependencies into containers, ensuring consistent behavior across environments. Use it to eliminate 'works on my machine' issues and simplify deployment."),
    ("How do I write good unit tests?", "Follow the AAA pattern: Arrange (set up test data), Act (call the function), Assert (check results). Test edge cases, keep tests independent, and aim for high coverage of critical paths."),
]

CODE_EXAMPLES = [
    # 데코레이터
    "# 파이썬 타이머 데코레이터\nimport time\nfrom functools import wraps\n\ndef timer(func):\n    @wraps(func)\n    def wrapper(*args, **kwargs):\n        start = time.perf_counter()\n        result = func(*args, **kwargs)\n        elapsed = time.perf_counter() - start\n        print(f'{func.__name__} 실행 시간: {elapsed:.4f}초')\n        return result\n    return wrapper\n\n@timer\ndef compute_sum(n):\n    return sum(range(n))\n\ncompute_sum(1_000_000)",
    # 이진 탐색
    "# 이진 탐색 (Binary Search)\ndef binary_search(arr, target):\n    left, right = 0, len(arr) - 1\n    while left <= right:\n        mid = (left + right) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            left = mid + 1\n        else:\n            right = mid - 1\n    return -1  # 미발견\n\nnums = list(range(0, 100, 2))\nprint(binary_search(nums, 42))  # 21",
    # 피보나치 제너레이터
    "# 제너레이터로 피보나치 수열\ndef fibonacci():\n    a, b = 0, 1\n    while True:\n        yield a\n        a, b = b, a + b\n\nfib = fibonacci()\nfirst_10 = [next(fib) for _ in range(10)]\nprint(first_10)  # [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]",
    # 클래스와 프로퍼티
    "# 파이썬 프로퍼티 활용\nclass Circle:\n    def __init__(self, radius):\n        self._radius = radius\n\n    @property\n    def radius(self):\n        return self._radius\n\n    @radius.setter\n    def radius(self, value):\n        if value < 0:\n            raise ValueError('반지름은 음수일 수 없습니다.')\n        self._radius = value\n\n    @property\n    def area(self):\n        import math\n        return math.pi * self._radius ** 2\n\nc = Circle(5)\nprint(f'넓이: {c.area:.2f}')  # 넓이: 78.54",
    # 컨텍스트 매니저
    "# 컨텍스트 매니저 직접 구현\nfrom contextlib import contextmanager\nimport time\n\n@contextmanager\ndef timer_context(label='작업'):\n    start = time.perf_counter()\n    try:\n        yield\n    finally:\n        elapsed = time.perf_counter() - start\n        print(f'{label} 완료: {elapsed:.4f}초')\n\nwith timer_context('데이터 처리'):\n    result = sorted(range(100000), reverse=True)",
    # 딕셔너리 컴프리헨션
    "# 딕셔너리 컴프리헨션 활용\nstudents = ['Alice', 'Bob', 'Charlie', 'Diana']\nscores = [92, 85, 78, 96]\n\n# zip으로 딕셔너리 생성\nscore_dict = {name: score for name, score in zip(students, scores)}\n\n# 조건부 필터링\npassed = {name: score for name, score in score_dict.items() if score >= 80}\nprint(passed)  # {'Alice': 92, 'Bob': 85, 'Diana': 96}",
    # 비동기 프로그래밍
    "# asyncio 기초 예시\nimport asyncio\n\nasync def fetch_data(url, delay):\n    print(f'{url} 요청 시작...')\n    await asyncio.sleep(delay)  # 네트워크 요청 시뮬레이션\n    print(f'{url} 응답 완료!')\n    return f'{url} 데이터'\n\nasync def main():\n    tasks = [\n        fetch_data('api/users', 1),\n        fetch_data('api/posts', 2),\n        fetch_data('api/comments', 0.5),\n    ]\n    results = await asyncio.gather(*tasks)\n    return results\n\nresults = asyncio.run(main())",
    # 데이터클래스
    "# dataclass 활용\nfrom dataclasses import dataclass, field\nfrom typing import List\n\n@dataclass\nclass Student:\n    name: str\n    age: int\n    grades: List[float] = field(default_factory=list)\n\n    @property\n    def average(self) -> float:\n        return sum(self.grades) / len(self.grades) if self.grades else 0.0\n\n    def __repr__(self):\n        return f'Student({self.name}, avg={self.average:.1f})'\n\ns = Student('김철수', 20, [90, 85, 92, 88])\nprint(s)  # Student(김철수, avg=88.8)",
    # 정렬 알고리즘
    "# 퀵 정렬 구현\ndef quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n    pivot = arr[len(arr) // 2]\n    left = [x for x in arr if x < pivot]\n    middle = [x for x in arr if x == pivot]\n    right = [x for x in arr if x > pivot]\n    return quicksort(left) + middle + quicksort(right)\n\ndata = [3, 6, 8, 10, 1, 2, 1]\nprint(quicksort(data))  # [1, 1, 2, 3, 6, 8, 10]",
    # LRU 캐시
    "# functools.lru_cache로 메모이제이션\nfrom functools import lru_cache\nimport time\n\n@lru_cache(maxsize=128)\ndef expensive_fib(n):\n    if n < 2:\n        return n\n    return expensive_fib(n - 1) + expensive_fib(n - 2)\n\nstart = time.time()\nprint(expensive_fib(50))  # 12586269025\nprint(f'계산 시간: {time.time()-start:.6f}초')\nprint(expensive_fib.cache_info())",
    # 파일 처리
    "# CSV 파일 읽기 및 처리\nimport csv\nfrom pathlib import Path\nfrom collections import defaultdict\n\ndef analyze_sales(filepath):\n    totals = defaultdict(float)\n    with open(filepath, encoding='utf-8', newline='') as f:\n        reader = csv.DictReader(f)\n        for row in reader:\n            product = row['product']\n            amount = float(row['amount'])\n            totals[product] += amount\n    return dict(sorted(totals.items(), key=lambda x: x[1], reverse=True))\n\n# 사용 예시\n# result = analyze_sales('sales.csv')\n# print(result)",
    # 정규 표현식
    "# 정규표현식 실용 예시\nimport re\n\n# 이메일 유효성 검사\ndef validate_email(email):\n    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$'\n    return bool(re.match(pattern, email))\n\n# 전화번호 정규화\ndef normalize_phone(phone):\n    digits = re.sub(r'\\D', '', phone)\n    if len(digits) == 11:\n        return f'{digits[:3]}-{digits[3:7]}-{digits[7:]}'\n    return phone\n\nprint(validate_email('test@example.com'))  # True\nprint(normalize_phone('010 1234 5678'))    # 010-1234-5678",
]


def generate_preference_data(max_workers: int = 2) -> list[dict]:
    """RLHF 선호 데이터 생성 (개선판) — 1 질문당 Ollama 호출 1회로 단순화.

    개선 사항:
      - 기존: 1질문당 generate×2 + judge×1 = 3호출 → 성공률 ~10%
      - 개선: 1질문당 단일 프롬프트(질문+자체채점) = 1호출 → 성공률 ~70%+
      - use_cache=False로 temperature 혼동 버그 제거
    """
    teacher = OllamaTeacher()
    if not teacher.is_available():
        logger.warning("⚠️  Ollama 미가동 — RLHF 데이터 건너뜀")
        return []

    questions = PREFERENCE_QUESTIONS * 3   # 40개 × 3 = 120 질문
    random.shuffle(questions)
    logger.info(f"🤖 RLHF 선호 데이터 생성 ({len(questions)}개 질문, 호출 1회/질문)...")

    def process_one(question: str):
        # 단일 프롬프트: 질문 → 모범 답변 직접 생성
        prompt = (
            f"다음 질문에 대해 사회인 수준의 명확하고 충실한 답변을 한국어로 작성하세요.\n"
            f"답변은 3문장 이상, 구체적인 내용을 포함하세요.\n\n"
            f"질문: {question}\n\n"
            f"답변:"
        )
        try:
            resp = teacher.generate(
                prompt,
                temperature=0.7,
                max_tokens=400,
                use_cache=True,   # 동일 질문 재시도 시 캐시 활용
                stage=6,
            )
            answer = resp.content.strip() if resp and resp.content else ""
            if len(answer) < 50:   # 너무 짧은 답변 제외
                return None
            return {
                "text": f"Q: {question}\nA: {answer}",
                "source": "preference_chosen",
                "stage": 6,
            }
        except Exception as e:
            logger.debug(f"RLHF 생성 실패 ({question[:30]}...): {e}")
            return None

    samples = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(process_one, q): q for q in questions}
        for i, future in enumerate(as_completed(futures), 1):
            result = future.result()
            if result:
                samples.append(result)
            if i % 10 == 0:
                logger.info(f"  진행: {i}/{len(questions)} | 선호 데이터: {len(samples)}건 "
                            f"(성공률: {len(samples)/i*100:.0f}%)")
    logger.info(f"✅ RLHF 선호 데이터: {len(samples):,}건 / {len(questions)}건 시도")
    return samples


def generate_conversation_data() -> list[dict]:
    """대화 형식 학습 데이터 (20개 템플릿 × 100회 반복 = 2,000건)."""
    samples = []
    for user_msg, assistant_msg in CONVERSATION_TEMPLATES:
        text = f"User: {user_msg}\nAssistant: {assistant_msg}"
        for _ in range(100):  # 20 → 100 (2,000건 확보)
            samples.append({"text": text, "source": "conversation", "stage": 6})
    logger.info(f"✅ 대화 데이터: {len(samples):,}건")
    return samples


def generate_code_data() -> list[dict]:
    samples = []
    for code in CODE_EXAMPLES:
        for _ in range(80):  # 15 → 80 (12개 × 80 = 960건)
            samples.append({"text": code, "source": "code_example", "stage": 6})
    logger.info(f"✅ 코드 데이터: {len(samples):,}건")
    return samples


def load_social_wiki(max_samples: int = 20000) -> list[dict]:
    """300자 이상 단락만 수집 (Stage 5 교훈 적용)."""
    samples = []
    try:
        from datasets import load_dataset
        ko_ds = load_dataset("wikimedia/wikipedia", "20231101.ko", split="train[:5000]")
        for item in ko_ds:
            if len(samples) >= max_samples: break
            for para in item.get("text","").strip().split("\n"):
                para = para.strip()
                if 300 <= len(para) <= 1000:
                    samples.append({"text": para, "source": "wiki_social", "stage": 6})
                if len(samples) >= max_samples: break
        random.shuffle(samples)
        logger.info(f"✅ 사회인 수준 Wikipedia: {len(samples):,}건")
    except Exception as e:
        logger.warning(f"⚠️  Wikipedia 실패: {e}")
    return samples


def load_sharegpt_social(target: int = 50000) -> list[dict]:
    """ShareGPT 한국어 심화 대화 — Stage 6 핵심 (자연스러운 대화 능력)."""
    import os, json as _json, glob as _glob
    samples = []
    cache_pattern = os.path.expanduser(
        "~/.cache/huggingface/hub/datasets--junelee--sharegpt_deepl_ko"
        "/snapshots/*/ko_dataset_2.json"
    )
    found = _glob.glob(cache_pattern)
    if not found:
        logger.info("📥 ShareGPT-ko 캐시 없음 — 다운로드 시도...")
        try:
            from huggingface_hub import hf_hub_download
            path = hf_hub_download(
                repo_id="junelee/sharegpt_deepl_ko",
                filename="ko_dataset_2.json",
                repo_type="dataset",
            )
            found = [path]
        except Exception as e:
            logger.warning(f"  ShareGPT-ko 다운로드 실패: {e}")
            return []
    logger.info(f"📥 ShareGPT-ko (사회인 심화) 로드: {found[0]}")
    with open(found[0], encoding="utf-8") as f:
        raw = _json.load(f)
    role_map = {"human": "User", "gpt": "Assistant", "user": "User", "assistant": "Assistant"}
    for item in raw:
        if len(samples) >= target: break
        convs = item.get("conversations", [])
        if len(convs) < 2: continue
        turns = []
        for c in convs[:8]:  # Stage 6: 더 긴 대화 허용
            role = role_map.get(c.get("from", "").lower())
            val  = c.get("value", "").strip().replace("\r", "")
            if role and val:
                turns.append(f"{role}: {val[:600]}")
        if len(turns) < 2: continue
        text = "\n".join(turns)
        if len(text) < 200 or len(text) > 2000: continue
        ko_ratio = sum(1 for ch in text if "\uAC00" <= ch <= "\uD7A3") / max(len(text), 1)
        if ko_ratio < 0.15: continue
        samples.append({"text": text, "source": "sharegpt_social", "stage": 6})
    logger.info(f"✅ ShareGPT 사회인 대화: {len(samples):,}건")
    return samples


def load_replay(max_per_stage: int = 1000) -> list[dict]:
    samples = []
    for path in PREV_STAGES:
        if not path.exists(): continue
        lines = [l for l in path.read_text().strip().split("\n") if l.strip()]
        random.shuffle(lines)
        n = min(max_per_stage, int(len(lines) * 0.05))
        for line in lines[:n]:
            try:
                d = json.loads(line)
                # 300자 미만 리플레이 제외
                if len(d.get("text", "")) < 300: continue
                d["source"] = f"{path.parent.name}_replay"
                d["stage"] = 6
                samples.append(d)
            except: pass
    logger.info(f"✅ 이전 단계 리플레이: {len(samples):,}건")
    return samples


def save_jsonl(samples, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for s in samples: f.write(json.dumps(s, ensure_ascii=False) + "\n")
    logger.info(f"💾 저장: {len(samples):,}건 → {path}")


def upload_wandb_artifact(train_path: Path, eval_path: Path, metadata_recipe: dict = None):
    """W&B 아티팩트 업로드 및 리니지 연결 (dataset-stage5 → dataset-stage6)."""
    try:
        import wandb, time
        run = wandb.init(
            project="EurekaAI-Curriculum",
            job_type="data_mixing",
            name="stage6_data_generation",
            config={"stage": 6, "source": "wiki+preference+conversation"},
            reinit=True,
        )
        # 이전 Stage 아티팩트 연결
        try:
            run.use_artifact("dataset-stage5:latest")
            logger.info("🔗 [W&B] 이전 dataset-stage5 연결 완료")
        except Exception as e:
            logger.warning(f"⚠️  dataset-stage5 연결 실패: {e}")

        art = wandb.Artifact("dataset-stage6", type="dataset",
                             description="Stage 6 Social dataset (preference+wiki+conversation)",
                             metadata=metadata_recipe or {})
        art.add_dir(str(STAGE_DIR))
        # 강제 버저닝용 dummy
        dummy = STAGE_DIR / "dummy_lineage.txt"
        dummy.write_text(f"lineage:{time.time()}")
        art.add_file(str(dummy))
        run.log_artifact(art)
        dummy.unlink(missing_ok=True)
        run.finish()
        logger.info("✅ [W&B] dataset-stage6 아티팩트 업로드 완료")
    except Exception as e:
        logger.warning(f"⚠️  W&B 업로드 실패: {e}")


def main():
    logger.info("=" * 60)
    logger.info("   EurekaAI — Stage 6: Social Data Preparation")
    logger.info("=" * 60)
    all_samples = []

    # 1. ShareGPT 심화 대화 (50,000건) — 자연스러운 대화 능력의 핵심
    logger.info("\n📥 [1/5] ShareGPT 사회인 심화 대화 수집 중...")
    all_samples += load_sharegpt_social(target=50000)

    # 2. Wikipedia 사회인 수준 단락 (20,000건)
    logger.info("\n📥 [2/5] Wikipedia 사회인 단락 수집 중...")
    all_samples += load_social_wiki(max_samples=20000)

    # 3. RLHF 선호 데이터 (Ollama Teacher) ★ Stage 6 핵심
    logger.info("\n📥 [3/5] RLHF 선호 데이터 생성 중...")
    all_samples += generate_preference_data(max_workers=2)

    # 4. 대화 템플릿 + 코드 예제 (시드)
    logger.info("\n📥 [4/5] 대화/코드 시드 데이터 생성 중...")
    all_samples += generate_conversation_data()
    all_samples += generate_code_data()

    # 5. 이전 Stage 리플레이
    logger.info("\n📥 [5/5] 이전 Stage 리플레이 수집 중...")
    all_samples += load_replay(max_per_stage=2000)

    seen = set()
    deduped = [s for s in all_samples if (k := s["text"][:80]) not in seen and not seen.add(k)]
    random.shuffle(deduped)

    from collections import Counter
    total = len(deduped)
    logger.info(f"\n📊 데이터 구성 ({total:,}건):")
    counts = Counter(s["source"] for s in deduped)
    for src, cnt in counts.most_common():
        logger.info(f"   {src:25s}: {cnt:6,} ({cnt/total*100:.1f}%)")

    n_eval = min(TARGET_EVAL, int(total * 0.1))
    save_jsonl(deduped[n_eval:], TRAIN_FILE)
    save_jsonl(deduped[:n_eval], EVAL_FILE)
    logger.info(f"\n✅ Stage 6 데이터 완료: Train {total-n_eval:,} / Eval {n_eval:,}")

    # 데이터셋 레시피 메타데이터 생성
    metadata_recipe = {
        "stage": 6,
        "total_samples": total,
        "train_samples": total - n_eval,
        "eval_samples": n_eval,
        "mixture_ratios": {
            src: f"{cnt/total*100:.1f}% ({cnt:,})" for src, cnt in counts.items()
        },
        "sampling_strategy": {
            "sharegpt_social": "length filtering, no random downsampling",
            "wiki_social": "300-1000 chars filtering",
            "replay": "random 2000 per previous stage"
        },
        "processing_details": {
            "deduplication": "exact match removed for first 80 chars"
        }
    }

    # Ollama 메모리 강제 언로드 (OOM 방지)
    try:
        import subprocess
        subprocess.run(["curl", "-s", "-X", "POST", "http://localhost:11434/api/generate",
                        "-d", '{"model": "gemma4:e4b", "keep_alive": 0}'], timeout=10)
        logger.info("🔌 Ollama 모델 언로드 완료 (메모리 확보)")
    except Exception:
        pass

    # W&B 리니지 업로드
    upload_wandb_artifact(TRAIN_FILE, EVAL_FILE, metadata_recipe=metadata_recipe)


if __name__ == "__main__":
    main()
