"""
EurekaAI — Ollama Teacher
Connects to a local Ollama instance to:
  1. Generate synthetic curriculum data per stage
  2. Create CoT (Chain-of-Thought) annotations
  3. Score student responses (reward signal)
  4. Generate Q&A pairs from passages
"""

import json
import time
import hashlib
import logging
from pathlib import Path
from typing import Optional, Union
from dataclasses import dataclass

import httpx

logger = logging.getLogger(__name__)


# ── Stage Prompts ───────────────────────────────────────────────────────────────

STAGE_CONTEXTS = {
    0: "신생아/유아 수준: 매우 짧고 단순한 문장만 사용. 반복적인 패턴. 단어 수 5개 이하.",
    1: "유아 수준: 간단한 단어와 일상 표현. 동화책 스타일. 문장당 10단어 이하.",
    2: "초등학생 수준: 쉬운 문법과 기초 지식. 질문-답변 형식 가능.",
    3: "중학생 수준: 논리적 설명, 단계적 추론. 수학/과학 기초 포함.",
    4: "고등학생 수준: 비판적 사고, 논증, 복잡한 추론. 에세이 스타일.",
    5: "대학생 수준: 전문 지식, 학술적 글쓰기, 복잡한 개념 설명.",
    6: "성인/사회인 수준: 자연스러운 대화, 뉴스, 코드, 복잡한 문제 해결.",
}

STAGE_NAMES_KO = {
    0: "신생아", 1: "유아", 2: "초등", 3: "중등",
    4: "고등", 5: "대학", 6: "사회인"
}


# ── Ollama Teacher ──────────────────────────────────────────────────────────────

@dataclass
class TeacherResponse:
    content: str
    model: str
    latency_ms: float
    cached: bool = False


class OllamaTeacher:
    """
    Teacher model wrapper using local Ollama API.
    Supports caching to avoid redundant API calls.
    """

    def __init__(
        self,
        model: str = "gemma4:e4b",
        base_url: str = "http://localhost:11434",
        timeout: float = 240.0,
        cache_dir: Optional[str] = "data/teacher_cache",
        max_retries: int = 3,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.cache_dir = Path(cache_dir) if cache_dir else None

        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        self._available: Optional[bool] = None

    # ── Health check ─────────────────────────────────────────────────────────

    def is_available(self) -> bool:
        """Check if Ollama server is running."""
        if self._available is not None:
            return self._available
        try:
            with httpx.Client(timeout=5.0) as client:
                resp = client.get(f"{self.base_url}/api/tags")
                self._available = resp.status_code == 200
        except Exception:
            self._available = False
        return self._available

    def list_models(self) -> list[str]:
        """List models available in Ollama."""
        try:
            with httpx.Client(timeout=10.0) as client:
                resp = client.get(f"{self.base_url}/api/tags")
                data = resp.json()
                return [m["name"] for m in data.get("models", [])]
        except Exception as e:
            logger.warning(f"Could not list Ollama models: {e}")
            return []

    # ── Core inference ────────────────────────────────────────────────────────

    def _cache_key(self, prompt: str) -> str:
        return hashlib.md5(f"{self.model}:{prompt}".encode()).hexdigest()

    def _load_cache(self, key: str, stage: Optional[int] = None) -> Optional[str]:
        if not self.cache_dir:
            return None
        
        base_path = self.cache_dir
        if stage is not None:
            base_path = base_path / f"stage_{stage}"
            
        path = base_path / f"{key}.json"
        if path.exists():
            return json.loads(path.read_text())["response"]
        return None

    def _save_cache(self, key: str, response: str, stage: Optional[int] = None):
        if not self.cache_dir:
            return
        
        base_path = self.cache_dir
        if stage is not None:
            base_path = base_path / f"stage_{stage}"
            base_path.mkdir(parents=True, exist_ok=True)
            
        path = base_path / f"{key}.json"
        path.write_text(json.dumps({"response": response}, ensure_ascii=False))

    def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 512,
        use_cache: bool = True,
        stage: Optional[int] = None,
    ) -> TeacherResponse:
        """Send a prompt to Ollama and return the response."""
        cache_key = self._cache_key(prompt + (system or ""))

        # Try cache first
        if use_cache:
            cached = self._load_cache(cache_key, stage=stage)
            if cached:
                return TeacherResponse(cached, self.model, 0.0, cached=True)

        if not self.is_available():
            raise RuntimeError(
                "❌ Ollama is not available. Please start it:\n"
                f"  ollama serve\n"
                f"  ollama pull {self.model}"
            )

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }

        for attempt in range(self.max_retries):
            try:
                t0 = time.time()
                with httpx.Client(timeout=self.timeout) as client:
                    resp = client.post(
                        f"{self.base_url}/api/chat",
                        json=payload,
                    )
                    resp.raise_for_status()
                    data = resp.json()
                    content = data["message"]["content"].strip()
                    latency = (time.time() - t0) * 1000

                if use_cache:
                    self._save_cache(cache_key, content, stage=stage)

                return TeacherResponse(content, self.model, latency)

            except httpx.TimeoutException:
                logger.warning(f"Ollama timeout (attempt {attempt+1}/{self.max_retries})")
                if attempt == self.max_retries - 1:
                    raise
                time.sleep(2 ** attempt)

    # ── Curriculum Data Generation ────────────────────────────────────────────

    def generate_qa_pairs(
        self,
        passage: str,
        stage: int = 2,
        n: int = 5,
    ) -> list[dict]:
        """
        Generate Q&A pairs from a passage, appropriate for the given stage.

        Returns:
            List of {"question": str, "answer": str, "stage": int}
        """
        stage_ctx = STAGE_CONTEXTS.get(stage, STAGE_CONTEXTS[2])
        prompt = (
            f"You are an educational AI creating {STAGE_NAMES_KO[stage]} level Q&A pairs.\n"
            f"Level: {stage_ctx}\n\n"
            f"다음 지문을 읽고 {n}개의 Q&A 쌍을 만들어주세요. 반드시 아래 형식의 JSON 배열(Array)로만 출력해야 합니다. 다른 사족은 절대 붙이지 마세요.\n"
            f"지문: {passage[:800]}\n\n"
            f"출력 예시:\n"
            f"[\n  {{\"question\": \"질문 내용\", \"answer\": \"답변 내용\"}}\n]\n\n"
            f"JSON 결괏값:"
        )

        resp = self.generate(prompt, temperature=0.6, max_tokens=2048, stage=stage)
        try:
            # Extract JSON from response
            text = resp.content
            start = text.find("[")
            end = text.rfind("]") + 1
            pairs = json.loads(text[start:end])
            return [{"stage": stage, **p} for p in pairs if "question" in p and "answer" in p]
        except Exception as e:
            logger.warning(f"Failed to parse QA pairs: {e}\nResponse: {resp.content[:200]}")
            return []

    def get_chain_of_thought(self, problem: str, stage: int = 3) -> str:
        """
        Generate a step-by-step reasoning trace for a math/logic problem.
        Used for Chain-of-Thought distillation.
        """
        prompt = (
            f"다음 문제를 풀이 과정(Chain-of-Thought)과 함께 자세히 설명해주세요.\n"
            f"대상은 {STAGE_NAMES_KO[stage]}학생이며, 명확하고 친절한 어투로 작성해주세요.\n\n"
            f"문제: {problem}"
        )
        resp = self.generate(prompt, temperature=0.6, max_tokens=1024, stage=stage)
        return resp.content

    def score_response(
        self,
        question: str,
        answer: str,
        reference: Optional[str] = None,
        stage: int = 2,
    ) -> float:
        """
        Score a student model response on a 0.0–1.0 scale.
        Uses heuristic scoring since Gemma4 thinking model returns empty content.
        Falls back to text depth analysis which is reliable for quality filtering.
        """
        import re

        # ── 1차: 텍스트 품질 휴리스틱 기반 채점 (LLM 불필요) ─────────────────
        # Gemma4 thinking 모델이 content를 비워두는 알려진 이슈를 우회
        score = self._heuristic_score(answer, stage)

        # ── 2차: LLM 채점 시도 (성공 시 덮어씀) ─────────────────────────────
        try:
            # yes/no 방식이 float 파싱보다 훨씬 안정적
            prompt = (
                f"Answer only 'yes' or 'no'.\n"
                f"Is this a high-quality university-level academic answer?\n"
                f"Answer: {answer[:600]}\n"
                f"High quality (yes/no):"
            )
            resp = self.generate(prompt, temperature=0.0, use_cache=False)
            content = resp.content.strip().lower()
            if content:
                if content.startswith("yes"):
                    score = max(score, 0.85)
                elif content.startswith("no"):
                    score = min(score, 0.45)
                else:
                    # float가 응답에 포함된 경우
                    m = re.search(r"0\.\d+|1\.0", content)
                    if m:
                        score = float(m.group(0))
        except Exception as e:
            logger.debug(f"LLM scoring failed, using heuristic: {e}")

        return max(0.0, min(1.0, score))

    def _heuristic_score(self, answer: str, stage: int) -> float:
        """텍스트 길이, 학술 키워드, 문장 복잡도 기반 품질 추정."""
        import re
        if not answer or len(answer) < 30:
            return 0.0

        score = 0.5  # base

        # 길이 기여 (stage가 높을수록 긴 답변 선호)
        min_len = {5: 150, 4: 100, 3: 80, 2: 50, 1: 20}.get(stage, 80)
        if len(answer) >= min_len * 2:
            score += 0.15
        elif len(answer) >= min_len:
            score += 0.05

        # 학술 키워드 (한/영 혼합)
        academic_kw = [
            "분석", "관점", "의미", "맥락", "함의", "학술", "이론", "개념",
            "논거", "주장", "근거", "결론", "따라서", "그러므로", "특히",
            "analysis", "perspective", "academic", "theoretical", "significant",
            "furthermore", "therefore", "consequently", "demonstrates", "implies"
        ]
        hits = sum(1 for kw in academic_kw if kw in answer)
        score += min(hits * 0.03, 0.20)

        # 문장 수 (다단락 구성)
        sentences = len(re.split(r'[.!?。]\s*', answer))
        if sentences >= 5:
            score += 0.10
        elif sentences >= 3:
            score += 0.05

        return min(score, 1.0)

    def generate_synthetic_stage_data(
        self,
        stage: int,
        n_samples: int = 50,
        language: str = "mixed",
    ) -> list[dict]:
        """
        Generate synthetic training data appropriate for a curriculum stage.

        Returns:
            List of {"text": str, "stage": int, "source": "synthetic"}
        """
        stage_ctx = STAGE_CONTEXTS.get(stage, STAGE_CONTEXTS[0])
        samples = []

        lang_instruction = {
            "ko": "한국어로만 작성하세요.",
            "en": "Write in English only.",
            "mixed": "한국어와 영어를 자유롭게 섞어서 작성하세요. Mix Korean and English freely.",
        }.get(language, "")

        batch_size = min(10, n_samples)
        batches = (n_samples + batch_size - 1) // batch_size

        for i in range(batches):
            current_n = min(batch_size, n_samples - i * batch_size)
            if current_n <= 0:
                break

            system = (
                f"You are generating training data for stage {stage} ({STAGE_NAMES_KO[stage]}) curriculum.\n"
                f"Level description: {stage_ctx}\n"
                f"{lang_instruction}\n"
                "Return valid JSON array of text samples."
            )
            prompt = (
                f"Create {current_n} educational text samples for {STAGE_NAMES_KO[stage]} level.\n"
                f"Each sample should be a short paragraph (50-200 chars).\n"
                f"JSON format: [{{\"text\": \"...\"}}]"
            )

            try:
                resp = self.generate(
                    prompt, system=system, temperature=0.8, max_tokens=2048, stage=stage
                )
                text = resp.content
                start = text.find("[")
                end = text.rfind("]") + 1
                parsed = json.loads(text[start:end])
                for item in parsed:
                    if "text" in item and item["text"].strip():
                        samples.append({
                            "text": item["text"].strip(),
                            "stage": stage,
                            "source": "synthetic_ollama",
                        })
            except Exception as e:
                logger.warning(f"Batch {i+1} synthetic data failed: {e}")

        logger.info(f"Generated {len(samples)} synthetic samples for stage {stage}")
        return samples

    def create_preference_pairs(
        self,
        question: str,
        answer_a: str,
        answer_b: str,
        stage: int = 6,
    ) -> dict:
        """
        Judge which of two answers is better — used for RLHF preference data.
        Returns: {"chosen": str, "rejected": str, "reason": str}
        """
        # yes/no 방식효으로 단순화 — float 파싱보다 안정적
        prompt = (
            f"Answer only 'A' or 'B'.\n"
            f"Which answer is better for the question?\n\n"
            f"Question: {question[:200]}\n"
            f"Answer A: {answer_a[:300]}\n"
            f"Answer B: {answer_b[:300]}\n\n"
            f"Better answer (A or B):"
        )
        try:
            resp = self.generate(prompt, temperature=0.0, use_cache=False, stage=stage)
            content = resp.content.strip().upper() if resp.content else ""
            if content.startswith("A"):
                winner = "A"
            elif content.startswith("B"):
                winner = "B"
            else:
                # heuristic fallback: 더 긴 답변이 더 충실한 답변
                winner = "A" if len(answer_a) >= len(answer_b) else "B"
        except Exception:
            winner = "A" if len(answer_a) >= len(answer_b) else "B"

        chosen   = answer_a if winner == "A" else answer_b
        rejected = answer_b if winner == "A" else answer_a
        return {"chosen": chosen, "rejected": rejected, "reason": f"winner={winner}"}
