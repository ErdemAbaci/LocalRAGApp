import re

import openai
from foundry_local import FoundryLocalManager

from app.config import MIN_GENERATIVE_ANSWER_CHARS


MODEL_ALIAS = "phi-4-mini"

ANSWER_STOP_MARKERS = [
    "Kaynak:",
    "kaynak:",
    "KAYNAK:",
    "Source:",
    "source:"
]

ANSWER_PREFIXES = [
    "Cevap:",
    "cevap:",
    "Answer:",
    "answer:"
]

CITATION_PATTERN = r"\[(?:Parça|parça|Parca|parca)[^\]]*\]"


def remove_answer_prefix(text):
    cleaned = text.strip()

    prefix_removed = True

    while prefix_removed:
        prefix_removed = False

        for prefix in ANSWER_PREFIXES:
            if cleaned.startswith(prefix):
                cleaned = cleaned[len(prefix):].strip()
                prefix_removed = True

    return cleaned


def clean_answer(answer):
    original_answer = answer.strip()
    filtered_lines = []

    for line in original_answer.splitlines():
        stripped = remove_answer_prefix(line)

        if not stripped:
            continue

        if stripped.lower().startswith(("kaynak:", "source:")):
            continue

        filtered_lines.append(stripped)

    cleaned = "\n".join(filtered_lines).strip()
    cleaned = re.sub(CITATION_PATTERN, "", cleaned).strip()

    for marker in ANSWER_STOP_MARKERS:
        marker_index = cleaned.find(marker)

        if marker_index > 0:
            cleaned = cleaned[:marker_index].strip()

    return cleaned or original_answer


def is_valid_answer(answer):
    if not answer:
        return False

    cleaned = answer.strip()

    if len(cleaned) < MIN_GENERATIVE_ANSWER_CHARS:
        return False

    if cleaned.lower().startswith(("kaynak:", "source:")):
        return False

    without_citations = re.sub(CITATION_PATTERN, "", cleaned).strip()
    without_prefix = remove_answer_prefix(without_citations)

    return len(without_prefix) >= MIN_GENERATIVE_ANSWER_CHARS


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
            temperature=0.1,
            max_tokens=220
        )

        raw_answer = response.choices[0].message.content

        return clean_answer(raw_answer)
