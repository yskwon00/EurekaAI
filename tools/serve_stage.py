#!/usr/bin/env python3
"""
EurekaAI Native Inference Server
=====================================
EurekaAI 체크포인트를 직접 로드하여 OpenAI 호환 API로 서빙합니다.
Ollama와 동일한 방식으로 curl / Open WebUI에서 사용 가능합니다.

사용법:
    python tools/serve_stage.py --stage 5
    python tools/serve_stage.py --stage 4 --port 11435
    python tools/serve_stage.py --list
    python tools/serve_stage.py --stage 5 --chat   # CLI 채팅

API 사용:
    curl http://localhost:11435/api/generate -d '{"model":"eureka-stage5","prompt":"안녕하세요"}'
    curl http://localhost:11435/v1/chat/completions -d '{...}'  # OpenAI 호환
"""

import argparse
import json
import sys
import threading
import time
import textwrap
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from core.model.config import EurekaConfig
from core.model.architecture import EurekaModel
from core.model.tokenizer_utils import EurekaTokenizer


# ---- Stage 메타데이터 -------------------------------------------------------
STAGE_META = {
    0: {"dir": "stage0_newborn",    "label": "신생아 (Newborn)"},
    1: {"dir": "stage1_toddler",    "label": "유아 (Toddler)"},
    2: {"dir": "stage2_elementary", "label": "초등 (Elementary)"},
    3: {"dir": "stage3_middle",     "label": "중등 (Middle)"},
    4: {"dir": "stage4_high",       "label": "고등 (High School)"},
    5: {"dir": "stage5_university", "label": "대학교 (University)"},
    6: {"dir": "stage6_social",     "label": "사회인 (Social)"},
}

STAGE_SYSTEM_PROMPTS = {
    0: "당신은 갓 태어난 아기 AI Eureka입니다. 매우 짧고 단순한 말만 사용합니다.",
    1: "당신은 유아 수준의 AI Eureka입니다. 쉽고 친근한 말로 짧게 대답합니다.",
    2: "당신은 초등학생 수준의 AI Eureka입니다. 쉬운 말로 친절하게 설명합니다.",
    3: "당신은 중학생 수준의 AI Eureka입니다. 논리적으로 단계별로 설명합니다.",
    4: "당신은 고등학생 수준의 AI Eureka입니다. 비판적 사고로 깊이 있게 답변합니다.",
    5: "당신은 대학생 수준의 AI Eureka입니다. 학술적이고 전문적인 언어로 깊이 있는 답변을 제공합니다.",
    6: "당신은 사회인 수준의 AI Eureka입니다. 전문 지식과 경험을 바탕으로 실용적인 답변을 제공합니다.",
}

CHECKPOINT_BASE = Path("checkpoints")
TOKENIZER_PATH  = Path("data/tokenizer/eureka.model")

# ---- 글로벌 모델 상태 --------------------------------------------------------
_model: EurekaModel | None     = None
_tokenizer: EurekaTokenizer | None = None
_config: EurekaConfig | None   = None
_stage: int                    = 5
_device: torch.device          = torch.device("cpu")


def find_checkpoint(stage: int, ckpt_type: str = "best") -> Path | None:
    meta = STAGE_META.get(stage)
    if not meta:
        return None
    d = meta["dir"]
    for candidate in [
        CHECKPOINT_BASE / d / d / ckpt_type,
        CHECKPOINT_BASE / d / ckpt_type,
    ]:
        if candidate.exists() and (candidate / "model.pt").exists():
            return candidate
    return None


def list_available():
    print("\n" + "=" * 62)
    print("  EurekaAI 사용 가능 체크포인트")
    print("=" * 62)
    found = False
    for stage, meta in STAGE_META.items():
        for ckpt_type in ["best", "final"]:
            path = find_checkpoint(stage, ckpt_type)
            if path:
                size_mb = sum(f.stat().st_size for f in path.rglob("*.pt")) / 1e6
                print(f"  --stage {stage}  [{ckpt_type:5s}]  {meta['label']:26s}  "
                      f"({size_mb:.1f}MB)")
                found = True
    if not found:
        print("  (체크포인트 없음 - 먼저 학습을 진행하세요)")
    print("=" * 62 + "\n")


def load_model(stage: int, ckpt_type: str = "best"):
    """모델과 토크나이저를 로드합니다."""
    global _model, _tokenizer, _config, _stage, _device

    ckpt_path = find_checkpoint(stage, ckpt_type)
    if not ckpt_path:
        raise FileNotFoundError(f"Stage {stage} [{ckpt_type}] 체크포인트를 찾을 수 없습니다.")

    print(f"📂 체크포인트 로드: {ckpt_path}")
    _config    = EurekaConfig.from_json(str(ckpt_path / "config.json"))
    _model     = EurekaModel(_config)
    state      = torch.load(ckpt_path / "model.pt", map_location="cpu", weights_only=True)
    _model.load_state_dict(state, strict=False)

    # 디바이스 설정 (MPS > CUDA > CPU)
    if torch.backends.mps.is_available():
        _device = torch.device("mps")
    elif torch.cuda.is_available():
        _device = torch.device("cuda")
    else:
        _device = torch.device("cpu")

    _model = _model.to(_device)
    _model.eval()
    _stage = stage

    _tokenizer = EurekaTokenizer(str(TOKENIZER_PATH))
    meta = STAGE_META[stage]
    print(f"✅ 모델 로드 완료: Stage {stage} {meta['label']}")
    print(f"   파라미터: {_model.num_parameters()/1e6:.2f}M  |  디바이스: {_device}")


# ---- 추론 -------------------------------------------------------------------

def generate_text(prompt: str, max_tokens: int = 200, temperature: float = 0.8,
                  top_k: int = 40, top_p: float = 0.9, repetition_penalty: float = 1.2) -> str:
    """텍스트를 생성합니다."""
    if _model is None or _tokenizer is None:
        return "(모델이 로드되지 않았습니다)"

    # 학습 데이터 형식에 맞게 프롬프트 구성
    # 위키 단락 or Q&A 형식 모두 지원
    if _stage >= 3:
        # "Q: {질문}\nA: " 형식 — 학습 데이터와 정확히 일치
        full_prompt = f"Q: {prompt}\nA: "
    else:
        full_prompt = prompt

    input_ids    = _tokenizer.encode(full_prompt)
    input_tensor = torch.tensor([input_ids], dtype=torch.long).to(_device)

    with torch.no_grad():
        output_ids = _model.generate(
            input_tensor,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
        )

    # 입력 이후 생성된 토큰만 추출
    new_ids = output_ids[0][len(input_ids):].tolist()
    decoded = _tokenizer.decode(new_ids).strip()

    # "Q:" 나 "\n" 이후 다음 대화 턴이 생성되면 첫 응답만 반환
    for stop_seq in ["\nQ:", "\n\nQ:", "Q: ", "\n---"]:
        if stop_seq in decoded:
            decoded = decoded.split(stop_seq)[0].strip()
            break

    return decoded


# ---- HTTP 서버 (Ollama 호환 API) --------------------------------------------

class EurekaHandler(BaseHTTPRequestHandler):
    """Ollama API 호환 HTTP 핸들러."""

    def log_message(self, format, *args):
        print(f"  [{self.client_address[0]}] {format % args}")

    def send_json(self, data: dict, status: int = 200):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        if self.path == "/api/tags":
            # Ollama 모델 목록
            self.send_json({
                "models": [{
                    "name": f"eureka-stage{_stage}",
                    "model": f"eureka-stage{_stage}",
                    "modified_at": "2026-04-21T00:00:00Z",
                    "size": _model.num_parameters() * 4 if _model else 0,
                    "details": {
                        "format": "pytorch",
                        "family": "EurekaAI",
                        "parameter_size": f"{_model.num_parameters()/1e6:.1f}M" if _model else "?",
                    }
                }]
            })
        elif self.path == "/":
            self.send_json({"status": "Eureka is running", "stage": _stage})
        else:
            self.send_json({"error": "Not found"}, 404)

    def do_POST(self):
        length  = int(self.headers.get("Content-Length", 0))
        body    = self.rfile.read(length)
        try:
            data = json.loads(body)
        except json.JSONDecodeError:
            self.send_json({"error": "Invalid JSON"}, 400)
            return

        # ── Ollama /api/generate 엔드포인트 ────────────────────────────────
        if self.path == "/api/generate":
            prompt      = data.get("prompt", "")
            opts        = data.get("options", {})
            max_tokens  = opts.get("num_predict", 200)
            temperature = opts.get("temperature", 0.8)
            rep_p       = opts.get("repetition_penalty", 1.2)

            t0       = time.time()
            response = generate_text(prompt, max_tokens=max_tokens, temperature=temperature, repetition_penalty=rep_p)
            elapsed  = time.time() - t0

            self.send_json({
                "model":       f"eureka-stage{_stage}",
                "created_at":  time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "response":    response,
                "done":        True,
                "total_duration": int(elapsed * 1e9),
            })

        # ── OpenAI 호환 /v1/chat/completions 엔드포인트 ─────────────────────
        elif self.path in ("/v1/chat/completions", "/api/chat"):
            messages    = data.get("messages", [])
            max_tokens  = data.get("max_tokens", 200)
            temperature = data.get("temperature", 0.8)

            # 메시지 → 프롬프트 변환
            prompt_parts = []
            for msg in messages:
                role    = msg.get("role", "user")
                content = msg.get("content", "")
                if role == "system":
                    prompt_parts.append(f"System: {content}")
                elif role == "user":
                    prompt_parts.append(f"User: {content}")
                elif role == "assistant":
                    prompt_parts.append(f"Eureka: {content}")
            prompt_parts.append("Eureka:")
            prompt = "\n".join(prompt_parts)

            t0       = time.time()
            response = generate_text(prompt, max_tokens=max_tokens, temperature=temperature)
            elapsed  = time.time() - t0

            self.send_json({
                "id":      f"chatcmpl-eureka-{int(t0)}",
                "object":  "chat.completion",
                "created": int(t0),
                "model":   f"eureka-stage{_stage}",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": response},
                    "finish_reason": "stop"
                }],
                "usage": {
                    "prompt_tokens": len(prompt.split()),
                    "completion_tokens": len(response.split()),
                    "total_tokens": len(prompt.split()) + len(response.split()),
                }
            })
        else:
            self.send_json({"error": f"Unknown endpoint: {self.path}"}, 404)


# ---- CLI 채팅 ---------------------------------------------------------------

def interactive_chat(stage: int):
    meta = STAGE_META[stage]
    print(f"\n{'='*55}")
    print(f"  EurekaAI Stage {stage} - {meta['label']}")
    print(f"  종료: 'quit' 또는 Ctrl+C")
    print(f"{'='*55}\n")

    while True:
        try:
            user_input = input("You: ").strip()
            if not user_input or user_input.lower() in ("quit", "exit", "종료"):
                break
            response = generate_text(user_input)
            print(f"\nEureka: {response}\n")
        except KeyboardInterrupt:
            break
    print("채팅을 종료합니다.")


# ---- 메인 -------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="EurekaAI Native Inference Server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            예시:
              python tools/serve_stage.py --list
              python tools/serve_stage.py --stage 5
              python tools/serve_stage.py --stage 5 --port 11435
              python tools/serve_stage.py --stage 5 --chat
              python tools/serve_stage.py --stage 5 --test "양자역학이란?"

            API 사용 (서버 시작 후):
              curl http://localhost:11435/api/generate \\
                -d '{"model":"eureka-stage5","prompt":"안녕하세요"}'

              curl http://localhost:11435/v1/chat/completions \\
                -H "Content-Type: application/json" \\
                -d '{"messages":[{"role":"user","content":"안녕하세요"}]}'
        """)
    )
    parser.add_argument("--stage",      type=int, choices=range(7),          help="서빙할 Stage 번호 (0~6)")
    parser.add_argument("--checkpoint", type=str, default="best",
                        choices=["best", "final"],                            help="체크포인트 타입 (기본: best)")
    parser.add_argument("--port",       type=int, default=11435,              help="서버 포트 (기본: 11435)")
    parser.add_argument("--host",       type=str, default="127.0.0.1",        help="서버 호스트 (기본: 127.0.0.1)")
    parser.add_argument("--list",       action="store_true",                  help="사용 가능한 체크포인트 목록 출력")
    parser.add_argument("--test",       type=str, default=None,               help="단발 테스트 후 종료")
    parser.add_argument("--chat",       action="store_true",                  help="인터랙티브 CLI 채팅")
    args = parser.parse_args()

    if args.list:
        list_available()
        return

    if args.stage is None:
        parser.print_help()
        return

    # 모델 로드
    print(f"\n🚀 EurekaAI Stage {args.stage} 서빙 준비 중...")
    load_model(args.stage, args.checkpoint)

    # 단발 테스트
    if args.test:
        print(f"\n입력: {args.test}")
        response = generate_text(args.test)
        print(f"응답: {response}")
        return

    # CLI 채팅
    if args.chat:
        interactive_chat(args.stage)
        return

    # HTTP 서버 시작
    server = HTTPServer((args.host, args.port), EurekaHandler)
    model_name = f"eureka-stage{args.stage}"
    print(f"\n{'='*55}")
    print(f"  ✅ EurekaAI Inference Server 시작!")
    print(f"  모델:   {model_name}  |  {STAGE_META[args.stage]['label']}")
    print(f"  주소:   http://{args.host}:{args.port}")
    print(f"  종료:   Ctrl+C")
    print(f"{'='*55}")
    print(f"\n  [Ollama 호환 API]")
    print(f"  curl http://{args.host}:{args.port}/api/generate \\")
    print(f'       -d \'{{"model":"{model_name}","prompt":"안녕하세요"}}\'')
    print(f"\n  [OpenAI 호환 API]")
    print(f"  curl http://{args.host}:{args.port}/v1/chat/completions \\")
    print(f"       -H 'Content-Type: application/json' \\")
    print(f'       -d \'{{"messages":[{{"role":"user","content":"안녕하세요"}}]}}\'')
    print()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n서버를 종료합니다.")
        server.shutdown()


if __name__ == "__main__":
    main()
