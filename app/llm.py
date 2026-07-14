import re
import subprocess
import time

import openai
from foundry_local import FoundryLocalManager

from app.config import MIN_GENERATIVE_ANSWER_CHARS


MODEL_ALIAS = "phi-4-mini"
FOUNDRY_START_ATTEMPTS = 100
FOUNDRY_START_INTERVAL_SECONDS = 0.1

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

CITATION_BODY_PATTERN = (
    r"(?:Parça|Parca)\s+\d+"
    r"(?:(?:\s*[-–,]\s*|\s+ve\s+)(?:(?:Parça|Parca)\s+)?\d+)*"
)
CITATION_PATTERN = rf"(?:\[{CITATION_BODY_PATTERN}\]|\({CITATION_BODY_PATTERN}\))"


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


def remove_citations(text):
    cleaned = re.sub(CITATION_PATTERN, "", text, flags=re.IGNORECASE)
    cleaned = re.sub(r"[ \t]+([.,;:!?])", r"\1", cleaned)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


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
    cleaned = remove_citations(cleaned)

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

    without_citations = remove_citations(cleaned)
    without_prefix = remove_answer_prefix(without_citations)

    return len(without_prefix) >= MIN_GENERATIVE_ANSWER_CHARS


def create_foundry_manager(show_startup_output=False):
    if show_startup_output:
        return FoundryLocalManager()

    manager = FoundryLocalManager(bootstrap=False)

    if manager.is_service_running():
        return manager

    with subprocess.Popen(
        ["foundry", "service", "start"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    ):
        for _ in range(FOUNDRY_START_ATTEMPTS):
            if manager.is_service_running():
                return manager

            time.sleep(FOUNDRY_START_INTERVAL_SECONDS)

    raise RuntimeError("Foundry Local servisi zamanında başlatılamadı.")


class LocalLLM:
    def __init__(self, show_startup_output=False):
        self.manager = create_foundry_manager(show_startup_output)

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
