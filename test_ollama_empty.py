import httpx

def test_ollama(prompt, system=None, temp=0.3):
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": "gemma4:e4b",
        "messages": messages,
        "stream": False,
        "options": {"temperature": temp}
    }
    print(f"Testing with system={bool(system)}, temp={temp}")
    try:
        resp = httpx.post("http://localhost:11434/api/chat", json=payload, timeout=60.0)
        res_json = resp.json()
        content = res_json.get("message", {}).get("content", "<NO_CONTENT>")
        print(f"-> Code {resp.status_code}, Res: {content[:100]}...\n")
    except Exception as e:
        print(f"-> Error: {e}\n")

test_ollama("2x + 5 = 13일 때 x를 구하세요. 단계별로 설명해주세요.", None, 0.3)
test_ollama("다음 문제를 단계별로 풀어주세요:\n\n2x + 5 = 13일 때 x를 구하세요.", "You are a 중학생 level teacher. Show reasoning step by step.", 0.3)
test_ollama("안녕", None, 0.7)
