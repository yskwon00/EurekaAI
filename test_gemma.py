import httpx, time
payload = {
    "model": "gemma4:e4b",
    "messages": [
        {"role": "system", "content": "You are a 중등 level teacher.\nShow your reasoning step by step before giving the final answer.\nUse both Korean and English naturally."},
        {"role": "user", "content": "다음 문제를 단계별로 풀어주세요:\n\n2x + 5 = 13일 때 x를 구하세요."}
    ],
    "stream": False,
    "options": {"temperature": 0.3, "num_predict": 512}
}
try:
    resp = httpx.post("http://localhost:11434/api/chat", json=payload, timeout=60.0)
    print("Status:", resp.status_code)
    print("Content:", repr(resp.json()["message"]["content"]))
except Exception as e:
    print("Error:", e)
