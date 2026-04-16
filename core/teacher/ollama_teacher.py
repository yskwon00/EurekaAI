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
        system = (
            f"You are an educational AI creating {STAGE_NAMES_KO[stage]} level Q&A pairs.\n"
            f"Level: {stage_ctx}\n"
            "Always respond in valid JSON array format."
        )
        prompt = (
            f"다음 지문을 읽고 {n}개의 Q&A 쌍을 만들어주세요.\n"
            f"지문: {passage[:800]}\n\n"
            f"JSON 형식으로 반환: [{{'question': '...', 'answer': '...'}}]"
        )

        resp = self.generate(prompt, system=system, temperature=0.6, max_tokens=1024, stage=stage)
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
        system = (
            f"You are a {STAGE_NAMES_KO[stage]} level teacher.\n"
            "Show your reasoning step by step before giving the final answer.\n"
            "Use both Korean and English naturally."
        )
        prompt = f"다음 문제를 단계별로 풀어주세요:\n\n{problem}"
        resp = self.generate(prompt, system=system, temperature=0.3, max_tokens=512, stage=stage)
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
        Used as a reward signal in RLVR/RLHF training.
        """
        system = (
            "You are an objective evaluator. "
            "Score the answer from 0.0 to 1.0. "
            "Return ONLY a float number, nothing else."
        )
        ref_text = f"\n정답 참고: {reference}" if reference else ""
        prompt = (
            f"질문: {question}\n"
            f"학생 답변: {answer}"
            f"{ref_text}\n\n"
            f"점수 (0.0~1.0):"
        )
        resp = self.generate(prompt, system=system, temperature=0.0, max_tokens=10, use_cache=False)
        try:
            score = float(resp.content.strip().split()[0])
            return max(0.0, min(1.0, score))
        except Exception:
            return 0.5  # neutral default

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

        Returns:
            {"chosen": str, "rejected": str, "reason": str}
        """
        system = (
            f"You are evaluating {STAGE_NAMES_KO[stage]} level AI responses.\n"
            "Choose which answer is better and explain briefly.\n"
            'Return JSON: {"winner": "A" or "B", "reason": "..."}'
        )
        prompt = (
            f"질문: {question}\n\n"
            f"답변 A: {answer_a}\n\n"
            f"답변 B: {answer_b}\n\n"
            "어느 답변이 더 나은가요?"
        )

        resp = self.generate(prompt, system=system, temperature=0.2, max_tokens=256, use_cache=False, stage=stage)
        try:
            text = resp.content
            start = text.find("{")
            end = text.rfind("}") + 1
            result = json.loads(text[start:end])
            winner = result.get("winner", "A")
            chosen = answer_a if winner == "A" else answer_b
            rejected = answer_b if winner == "A" else answer_a
            return {"chosen": chosen, "rejected": rejected, "reason": result.get("reason", "")}
        except Exception:
            return {"chosen": answer_a, "rejected": answer_b, "reason": "parse_error"}
