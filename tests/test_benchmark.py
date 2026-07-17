import json
import tempfile
import unittest
from pathlib import Path

from app.benchmark import (
    BenchmarkPreparationError,
    normalize_model_aliases,
    run_model_benchmark,
)


class IncrementingTimer:
    def __init__(self):
        self.value = 0.0

    def __call__(self):
        current = self.value
        self.value += 1.0
        return current


class FakeLLM:
    loaded_models = []

    def __init__(self, model_alias, show_startup_output):
        self.model_alias = model_alias
        self.show_startup_output = show_startup_output
        self.loaded_models.append(model_alias)

    def generate_answer(self, messages):
        question = messages[-1]["content"]

        if "temel hedefi" in question:
            return "Bilgi güvenliğinin hedefleri gizlilik, bütünlük ve erişilebilirliktir."

        return "Gönderen kontrol edilmeli, bağlantı incelenmeli ve çok faktörlü doğrulama kullanılmalıdır."


def fake_retrieval(question, top_k):
    chunk_text = (
        "Bilgi güvenliğinin hedefleri gizlilik, bütünlük ve erişilebilirliktir."
        if "temel hedefi" in question
        else "Gönderen ve bağlantı kontrol edilir, çok faktörlü doğrulama kullanılır."
    )
    return [{
        "id": 1,
        "source_name": "cybersecurity.txt",
        "source_type": "txt",
        "page_number": None,
        "chunk_index": 1,
        "chunk_text": chunk_text,
        "score": 0.75,
    }]


class BenchmarkTests(unittest.TestCase):
    def setUp(self):
        FakeLLM.loaded_models = []

    def test_benchmark_writes_cold_warm_and_quality_results(self):
        cases = [
            {
                "name": "security_goals",
                "question": "Bilgi güvenliğinin üç temel hedefi nelerdir?",
                "expected_source": "cybersecurity.txt",
                "min_score": 0.40,
                "expected_terms": ["gizlilik", "bütünlük", "erişilebilirlik"],
            },
            {
                "name": "phishing",
                "question": "Kimlik avından nasıl korunulur?",
                "expected_source": "cybersecurity.txt",
                "expected_terms": ["gönderen", "bağlantı", "çok faktörlü"],
            },
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            cases_path = root / "cases.json"
            report_path = root / "report.json"
            cases_path.write_text(json.dumps(cases), encoding="utf-8")
            report, written_path = run_model_benchmark(
                ["phi-4-mini", "phi-4-mini"],
                cases_path=cases_path,
                report_path=report_path,
                llm_factory=FakeLLM,
                retrieval_func=fake_retrieval,
                timer=IncrementingTimer(),
            )
            stored_report = json.loads(report_path.read_text(encoding="utf-8"))

        self.assertEqual(FakeLLM.loaded_models, ["phi-4-mini"])
        self.assertEqual(written_path, report_path)
        self.assertEqual(report["models"][0]["status"], "ok")
        self.assertEqual(report["models"][0]["summary"]["valid_case_count"], 2)
        self.assertEqual(
            report["models"][0]["summary"]["average_term_coverage"],
            1.0,
        )
        self.assertEqual(
            [run["label"] for run in report["models"][0]["cases"][0]["runs"]],
            ["cold", "warm"],
        )
        self.assertEqual(stored_report["case_count"], 2)

    def test_model_load_error_is_recorded_without_losing_report(self):
        cases = [{
            "name": "security_goals",
            "question": "Bilgi güvenliğinin üç temel hedefi nelerdir?",
            "expected_source": "cybersecurity.txt",
            "expected_terms": [],
        }]

        def failing_factory(**kwargs):
            raise RuntimeError("model cache bulunamadı")

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            cases_path = root / "cases.json"
            report_path = root / "report.json"
            cases_path.write_text(json.dumps(cases), encoding="utf-8")
            report, _ = run_model_benchmark(
                ["missing-model"],
                cases_path=cases_path,
                report_path=report_path,
                llm_factory=failing_factory,
                retrieval_func=fake_retrieval,
                timer=IncrementingTimer(),
            )

        result = report["models"][0]
        self.assertEqual(result["status"], "error")
        self.assertIn("model cache bulunamadı", result["error"])

    def test_wrong_retrieval_source_stops_benchmark(self):
        cases = [{
            "name": "wrong_source",
            "question": "Bilgi güvenliği nedir?",
            "expected_source": "expected.txt",
        }]

        with tempfile.TemporaryDirectory() as temp_dir:
            cases_path = Path(temp_dir) / "cases.json"
            cases_path.write_text(json.dumps(cases), encoding="utf-8")

            with self.assertRaisesRegex(BenchmarkPreparationError, "yanlış kaynağı"):
                run_model_benchmark(
                    ["phi-4-mini"],
                    cases_path=cases_path,
                    report_path=Path(temp_dir) / "report.json",
                    llm_factory=FakeLLM,
                    retrieval_func=fake_retrieval,
                )

    def test_model_aliases_are_trimmed_and_deduplicated(self):
        self.assertEqual(
            normalize_model_aliases([" phi-4-mini ", "phi-4-mini", "phi-3.5-mini"]),
            ["phi-4-mini", "phi-3.5-mini"],
        )


if __name__ == "__main__":
    unittest.main()
