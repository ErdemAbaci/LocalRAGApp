import io
import tomllib
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import main
from app.index_state import IndexFreshness


class CliEntrypointTests(unittest.TestCase):
    def tearDown(self):
        main.DEBUG = False
        main._llm = None

    def test_no_subcommand_starts_interactive_cli(self):
        with patch("main.main") as interactive_cli:
            exit_code = main.cli([])

        self.assertEqual(exit_code, 0)
        interactive_cli.assert_called_once_with()

    def test_ask_joins_question_and_returns_success(self):
        with patch("main.answer_question", return_value=True) as answer_question:
            exit_code = main.cli(["ask", "RAG", "nedir?"])

        self.assertEqual(exit_code, 0)
        answer_question.assert_called_once_with("RAG nedir?")

    def test_ask_failure_returns_nonzero_exit_code(self):
        with patch("main.answer_question", return_value=False):
            exit_code = main.cli(["ask", "boş indeks"])

        self.assertEqual(exit_code, 1)

    def test_named_subcommands_use_shared_command_runner(self):
        commands = ["reindex", "stats", "sources", "doctor", "model", "config"]

        for command in commands:
            with self.subTest(command=command):
                with patch("main.execute_command", return_value=True) as execute:
                    exit_code = main.cli([command])

                self.assertEqual(exit_code, 0)
                execute.assert_called_once_with(f"/{command}")

    def test_debug_flag_is_forwarded_to_shared_rag_flow(self):
        with patch("main.answer_question", return_value=True):
            exit_code = main.cli(["--debug", "ask", "RAG nedir?"])

        self.assertEqual(exit_code, 0)
        self.assertTrue(main.DEBUG)

    def test_benchmark_forwards_selected_models(self):
        with patch("main.run_benchmark_command", return_value=True) as benchmark:
            exit_code = main.cli([
                "benchmark",
                "--models",
                "phi-4-mini",
                "phi-3.5-mini",
            ])

        self.assertEqual(exit_code, 0)
        benchmark.assert_called_once_with(["phi-4-mini", "phi-3.5-mini"])

    def test_interactive_benchmark_uses_active_model_by_default(self):
        with patch("main.run_benchmark_command", return_value=True) as benchmark:
            result = main.handle_command("/benchmark")

        self.assertEqual(result, "handled")
        benchmark.assert_called_once_with([main.MODEL_ALIAS])

    def test_no_evidence_answer_is_a_successful_cli_result(self):
        chunks = [{
            "id": 1,
            "source_name": "example.txt",
            "chunk_text": "RAG açıklaması",
            "score": 0.05,
        }]
        buffer = io.StringIO()

        with patch("main.get_index_freshness", return_value=IndexFreshness("current")):
            with patch("main.get_top_chunks", return_value=chunks):
                with redirect_stdout(buffer):
                    success = main.answer_question("Hava nasıl?")

        self.assertTrue(success)
        self.assertIn("Bu bilgi verilen dokümanlarda yok.", buffer.getvalue())

    def test_extractive_answer_uses_shared_question_flow_without_llm(self):
        chunks = [{
            "id": 1,
            "source_name": "example.txt",
            "source_type": "txt",
            "page_number": None,
            "chunk_index": 1,
            "chunk_text": "RAG, retrieval ve generation adımlarını birleştirir.",
            "score": 0.70,
        }]
        buffer = io.StringIO()

        with patch("main.get_index_freshness", return_value=IndexFreshness("current")):
            with patch("main.get_top_chunks", return_value=chunks):
                with patch("main.get_llm") as get_llm:
                    with redirect_stdout(buffer):
                        success = main.answer_question("RAG nedir?")

        self.assertTrue(success)
        self.assertIn("Doğrudan", buffer.getvalue())
        get_llm.assert_not_called()

    def test_stale_index_warns_but_still_answers(self):
        chunks = [{
            "id": 1,
            "source_name": "example.txt",
            "chunk_text": "RAG açıklaması",
            "score": 0.05,
        }]
        stale = IndexFreshness(status="stale", modified=("example.txt",))
        buffer = io.StringIO()

        with patch("main.get_index_freshness", return_value=stale):
            with patch("main.get_top_chunks", return_value=chunks):
                with redirect_stdout(buffer):
                    success = main.answer_question("Hava nasıl?")

        output = buffer.getvalue()
        self.assertTrue(success)
        self.assertIn("İndeks güncel değil", output)
        self.assertIn("Değişen: example.txt", output)
        self.assertIn("local-rag reindex", output)
        self.assertIn("Bu bilgi verilen dokümanlarda yok.", output)

    def test_pyproject_registers_local_rag_console_script(self):
        project_root = Path(__file__).resolve().parents[1]
        metadata = tomllib.loads(
            (project_root / "pyproject.toml").read_text(encoding="utf-8")
        )

        self.assertEqual(metadata["project"]["requires-python"], ">=3.11")
        self.assertEqual(metadata["project"]["scripts"]["local-rag"], "main:cli")
        self.assertEqual(
            metadata["tool"]["setuptools"]["dynamic"]["version"]["attr"],
            "app.__version__",
        )

    def test_help_uses_turkish_cli_labels(self):
        buffer = io.StringIO()

        with redirect_stdout(buffer):
            with self.assertRaisesRegex(SystemExit, "0"):
                main.cli(["--help"])

        output = buffer.getvalue()
        self.assertIn("kullanım:", output)
        self.assertIn("seçenekler:", output)
        self.assertIn("komutlar:", output)
        self.assertIn("Bu yardım metnini gösterir", output)
        self.assertIn("add", output)
        self.assertIn("remove", output)
        self.assertIn("benchmark", output)


if __name__ == "__main__":
    unittest.main()
