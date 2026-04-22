import sys
from pathlib import Path
sys.path.insert(0, str(Path("/Users/yskwon00/MyWorkspace/EurekaAI")))
from core.teacher.ollama_teacher import OllamaTeacher

t = OllamaTeacher()
q = "1999년 국가안전기획부(안기부)가 국가정보원(국정원)으로 명칭이 변경되고 부총리급에서 장관급으로 직급이 조정된 배경을 제도적 측면에서 분석하고, 이러한 국가 정보기관의 제도화 과정이 대한민국 안보 시스템의 전문성과 책임성 강화에 기여한 학술적 의미는 무엇인지 서술하시오."
a = "안기부의 국정원 개편은 냉전 종식 이후 변화하는 안보 환경과 정보전의 복잡성에 대응하기 위한 제도적 정비 과정으로 해석된다. 학술적으로 볼 때, 이는 정보 수집 및 분석 기능을 고도의 전문성을 요구하는 장관급 기관으로 격상시켜 국가 안보의 핵심 축으로 자리매김하게 했음을 의미한다. 특히, 단순한 정치적 필요성을 넘어, 정보 수집의 독립성과 전문적 역량을 제도적으로 보장함으로써 정보 활동의 투명성과 책임성을 강화하는 방향으로 진화했음을 보여준다. 이는 국가 안보 시스템이 시대적 요구에 따라 끊임없이 재구조화되는 학술적 사례로 평가된다."

prompt = (
    f"You are an objective academic evaluator.\n"
    f"Evaluate the generated answer based on logic, depth, and clarity for University level.\n"
    f"질문: {q}\n"
    f"학생 답변: {a}\n\n"
    f"Score the answer from 0.0 to 1.0 (e.g., 0.8 or 0.95).\n"
    f"Return ONLY a float number, nothing else:\n"
)
resp = t.generate(prompt, temperature=0.1, use_cache=False)
print("---RAW OUTPUT---")
print(repr(resp.content))
print("----------------")
