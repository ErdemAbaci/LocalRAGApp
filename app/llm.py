import openai
from foundry_local import FoundryLocalManager


MODEL_ALIAS = "phi-3.5-mini"


def clean_answer(answer):
    cleaned = answer.strip()

    prefixes = [
        "Cevap:",
        "cevap:",
        "Answer:",
        "answer:"
    ]

    for prefix in prefixes:
        if cleaned.startswith(prefix):
            cleaned = cleaned[len(prefix):].strip()

    lines = cleaned.splitlines()
    filtered_lines = []

    for line in lines:
        stripped = line.strip()

        if stripped.lower().startswith("kaynak:"):
            continue

        filtered_lines.append(line)

    return "\n".join(filtered_lines).strip()


class LocalLLM:
    def __init__(self):
        print("Foundry Local başlatılıyor...")
        self.manager = FoundryLocalManager()

        print(f"Model yükleniyor: {MODEL_ALIAS}")
        self.model_info = self.manager.load_model(MODEL_ALIAS)

        self.client = openai.OpenAI(
            base_url=self.manager.endpoint,
            api_key=self.manager.api_key
        )

    def generate_answer(self, messages):
        response = self.client.chat.completions.create(
            model=self.model_info.id,
            messages=messages,
            temperature=0.1
        )

        raw_answer = response.choices[0].message.content

        return clean_answer(raw_answer)