import os
import re
import subprocess
import time
from collections import Counter
from contextlib import contextmanager

import foundry_local.api as foundry_api
import openai
from foundry_local import FoundryLocalManager

from app.config import MIN_GENERATIVE_ANSWER_CHARS


DEFAULT_MODEL_ALIAS = "phi-4-mini"
MODEL_ALIAS_ENV_VAR = "LOCAL_RAG_MODEL"


def get_model_alias(environ=None):
    environment = os.environ if environ is None else environ
    configured_alias = environment.get(MODEL_ALIAS_ENV_VAR, "").strip()
    return configured_alias or DEFAULT_MODEL_ALIAS


def get_model_alias_source(environ=None):
    environment = os.environ if environ is None else environ
    return (
        MODEL_ALIAS_ENV_VAR
        if environment.get(MODEL_ALIAS_ENV_VAR, "").strip()
        else "varsayılan"
    )


MODEL_ALIAS = get_model_alias()
FOUNDRY_START_ATTEMPTS = 100
FOUNDRY_START_INTERVAL_SECONDS = 0.1
FOUNDRY_STATUS_TIMEOUT_SECONDS = 15
FOUNDRY_HTTP_TIMEOUT_SECONDS = 120

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


def has_excessive_repetition(answer):
    words = re.findall(r"\b\w+\b", answer.casefold(), flags=re.UNICODE)

    if len(words) < 12:
        return False

    word_counts = Counter(words)
    most_common_count = word_counts.most_common(1)[0][1]

    if most_common_count >= 8 and most_common_count / len(words) > 0.25:
        return True

    trigrams = list(zip(words, words[1:], words[2:]))
    trigram_counts = Counter(trigrams)
    return bool(trigram_counts and trigram_counts.most_common(1)[0][1] >= 3)


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

    if has_excessive_repetition(without_prefix):
        return False

    return len(without_prefix) >= MIN_GENERATIVE_ANSWER_CHARS


def get_foundry_service_uri():
    try:
        result = subprocess.run(
            ["foundry", "service", "status"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=FOUNDRY_STATUS_TIMEOUT_SECONDS,
            check=False,
        )
    except subprocess.TimeoutExpired as error:
        raise RuntimeError(
            "Foundry Local servis durumu zamanında alınamadı."
        ) from error

    match = re.search(
        r"http://(?:[a-zA-Z0-9.-]+|\d{1,3}(?:\.\d{1,3}){3}):\d+",
        result.stdout,
    )
    return match.group(0) if match else None


@contextmanager
def safe_foundry_service_lookup():
    original_lookup = foundry_api.get_service_uri
    foundry_api.get_service_uri = get_foundry_service_uri

    try:
        yield
    finally:
        foundry_api.get_service_uri = original_lookup


def create_foundry_manager(show_startup_output=False):
    with safe_foundry_service_lookup():
        manager = FoundryLocalManager(
            bootstrap=False,
            timeout=FOUNDRY_HTTP_TIMEOUT_SECONDS,
        )

        if manager.is_service_running():
            return manager

        process_options = {}

        if not show_startup_output:
            process_options = {
                "stdout": subprocess.DEVNULL,
                "stderr": subprocess.DEVNULL,
            }

        with subprocess.Popen(
            ["foundry", "service", "start"],
            **process_options,
        ):
            for _ in range(FOUNDRY_START_ATTEMPTS):
                if manager.is_service_running():
                    return manager

                time.sleep(FOUNDRY_START_INTERVAL_SECONDS)

    raise RuntimeError("Foundry Local servisi zamanında başlatılamadı.")


class LocalLLM:
    def __init__(self, show_startup_output=False, model_alias=None):
        self.model_alias = model_alias or MODEL_ALIAS
        self.manager = create_foundry_manager(show_startup_output)

        self.model_info = self.manager.load_model(self.model_alias)

        self.client = openai.OpenAI(
            base_url=self.manager.endpoint,
            api_key=self.manager.api_key,
            timeout=FOUNDRY_HTTP_TIMEOUT_SECONDS,
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
