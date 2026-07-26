import io
import unittest
from contextlib import redirect_stdout
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import main
from app.cli_output import (
    ANSWER_MODE_STYLES,
    MUTED_AMBER,
    MUTED_GREEN,
    PLAIN_INPUT_PROMPT,
    PRIMARY_BRIGHT,
    READLINE_INPUT_PROMPT,
    RAGProgress,
    print_answer,
    print_issue,
    print_performance,
    read_prompt,
)
from app.index_state import IndexFreshness


class CliOutputTests(unittest.TestCase):
    def tearDown(self):
        main.DEBUG = False
        main._llm = None
        main._source_filter = None

    def test_print_issue_hides_technical_detail_by_default(self):
        buffer = io.StringIO()

        with redirect_stdout(buffer):
            print_issue(
                "error",
                "İşlem başarısız.",
                solution="/doctor çalıştır.",
                error=RuntimeError("gizli teknik ayrıntı"),
            )

        output = buffer.getvalue()
        self.assertIn("HATA", output)
        self.assertIn("İşlem başarısız.", output)
        self.assertIn("Çözüm", output)
        self.assertIn("/doctor çalıştır.", output)
        self.assertNotIn("gizli teknik ayrıntı", output)

    def test_read_prompt_is_registered_with_readline_in_terminal(self):
        fake_console = SimpleNamespace(
            is_terminal=True,
            color_system="truecolor",
            print=MagicMock(),
        )

        with patch("app.cli_output.console", fake_console):
            with patch("app.cli_output.sys.stdin.isatty", return_value=True):
                with patch("builtins.input", return_value="/sources") as input_func:
                    result = read_prompt()

        self.assertEqual(result, "/sources")
        fake_console.print.assert_called_once_with()
        input_func.assert_called_once_with(READLINE_INPUT_PROMPT)

    def test_read_prompt_uses_plain_text_without_tty(self):
        fake_console = SimpleNamespace(
            is_terminal=False,
            color_system=None,
            print=MagicMock(),
        )

        with patch("app.cli_output.console", fake_console):
            with patch("builtins.input", return_value="RAG nedir?") as input_func:
                result = read_prompt()

        self.assertEqual(result, "RAG nedir?")
        self.assertEqual(PLAIN_INPUT_PROMPT, "> ")
        input_func.assert_called_once_with(PLAIN_INPUT_PROMPT)

    def test_print_issue_shows_technical_detail_in_debug_mode(self):
        buffer = io.StringIO()

        with redirect_stdout(buffer):
            print_issue(
                "error",
                "İşlem başarısız.",
                error=RuntimeError("bağlantı reddedildi"),
                debug=True,
            )

        self.assertIn(
            "RuntimeError: bağlantı reddedildi",
            buffer.getvalue(),
        )

    def test_reindex_error_uses_standard_message_and_preserves_session(self):
        buffer = io.StringIO()

        with patch("main.ingest_documents", side_effect=ValueError("bozuk pdf")):
            with redirect_stdout(buffer):
                result = main.handle_command("/reindex")

        output = buffer.getvalue()
        self.assertEqual(result, "handled")
        self.assertIn("HATA", output)
        self.assertIn("Re-index tamamlanamadı", output)
        self.assertIn("mevcut indeks korundu", output)
        self.assertIn("Çözüm", output)
        self.assertNotIn("bozuk pdf", output)

    def test_command_error_shows_detail_when_debug_is_enabled(self):
        main.DEBUG = True
        buffer = io.StringIO()

        with patch("main.print_stats", side_effect=RuntimeError("sqlite kilitli")):
            with redirect_stdout(buffer):
                result = main.handle_command("/stats")

        output = buffer.getvalue()
        self.assertEqual(result, "handled")
        self.assertIn("HATA", output)
        self.assertIn("Sistem bilgileri okunamadı.", output)
        self.assertIn("RuntimeError: sqlite kilitli", output)

    def test_retrieval_error_does_not_close_cli_session(self):
        buffer = io.StringIO()

        with patch("builtins.input", side_effect=["RAG nedir?", "/exit"]):
            with patch("main.get_top_chunks", side_effect=RuntimeError("embedding bozuk")):
                with redirect_stdout(buffer):
                    main.main()

        output = buffer.getvalue()
        self.assertIn("HATA", output)
        self.assertIn("Dokümanlarda arama yapılamadı.", output)
        self.assertIn("/doctor çalıştır", output)
        self.assertNotIn("embedding bozuk", output)

    def test_answer_and_performance_render_key_information(self):
        buffer = io.StringIO()

        with redirect_stdout(buffer):
            print_answer("RAG, retrieval ve generation kullanır.", "generative", 0.6123)
            print_performance(0.08, 4.2, 4.28)

        output = buffer.getvalue()
        self.assertIn("Cevap", output)
        self.assertIn("RAG, retrieval ve generation kullanır.", output)
        self.assertIn("Üretken", output)
        self.assertNotIn("generative", output)
        self.assertIn("Skor", output)
        self.assertIn("0.6123", output)
        self.assertIn("Arama", output)
        self.assertIn("Yanıt", output)
        self.assertIn("Toplam", output)
        self.assertIn("4.280 sn", output)

    def test_rag_progress_reuses_one_status_for_all_stages(self):
        status = MagicMock()
        fake_console = SimpleNamespace(
            is_terminal=True,
            status=MagicMock(return_value=status),
        )
        progress = RAGProgress("phi-4-mini", console_instance=fake_console)

        with progress:
            with progress.stage("retrieval"):
                pass
            with progress.stage("model"):
                pass
            with progress.stage("generation"):
                pass

        fake_console.status.assert_called_once()
        status.start.assert_called_once_with()
        status.stop.assert_called_once_with()
        self.assertGreaterEqual(status.update.call_count, 5)
        self.assertEqual(
            progress.render().plain,
            "✓ Arama  ·  ✓ phi-4-mini  ·  ✓ Yanıt",
        )

    def test_rag_progress_switches_from_status_to_transient_answer(self):
        status = MagicMock()
        live = MagicMock()
        live_factory = MagicMock(return_value=live)
        fake_console = SimpleNamespace(
            is_terminal=True,
            status=MagicMock(return_value=status),
        )
        progress = RAGProgress(
            "phi-4-mini",
            console_instance=fake_console,
            live_factory=live_factory,
        )

        with progress:
            with progress.stage("retrieval"):
                pass
            with progress.stage("model"):
                pass
            with progress.stage("generation"):
                progress.update_answer("İlk cevap parçası")
                progress.update_answer("Tam cevap metni")

        status.stop.assert_called_once_with()
        live.start.assert_called_once_with(refresh=True)
        live.update.assert_called_once()
        live.stop.assert_called_once_with()
        self.assertTrue(live_factory.call_args.kwargs["transient"])

    def test_answer_modes_have_distinct_turkish_labels_and_styles(self):
        expected_modes = {
            "generative": ("Üretken", PRIMARY_BRIGHT),
            "extractive": ("Doğrudan", MUTED_GREEN),
            "fallback_extractive": ("Kaynak metni", MUTED_AMBER),
            "no_evidence": ("Kanıt bulunamadı", "bright_black"),
        }

        self.assertEqual(ANSWER_MODE_STYLES, expected_modes)

        for mode, (label, _) in expected_modes.items():
            with self.subTest(mode=mode):
                buffer = io.StringIO()

                with redirect_stdout(buffer):
                    print_answer("Örnek cevap.", mode, 0.42)

                output = buffer.getvalue()
                self.assertIn(label, output)
                self.assertNotIn(mode, output)

    def test_successful_reindex_reports_chunk_count(self):
        buffer = io.StringIO()

        with patch("main.ingest_documents", return_value=12):
            with redirect_stdout(buffer):
                result = main.handle_command("/reindex")

        output = buffer.getvalue()
        self.assertEqual(result, "handled")
        self.assertIn("OK", output)
        self.assertIn("Re-index tamamlandı", output)
        self.assertIn("12 chunk", output)

    def test_model_command_is_read_only_and_shows_lazy_load_state(self):
        checks = [
            SimpleNamespace(
                name="Foundry Local",
                status="ok",
                message="Terminal aracı hazır.",
                solution=None,
            ),
            SimpleNamespace(
                name="LLM modeli",
                status="ok",
                message="phi-4-mini cache içinde hazır.",
                solution=None,
            ),
        ]
        buffer = io.StringIO()

        with patch("main.check_foundry", return_value=checks):
            with patch("main.get_local_model_path", return_value="/cache/embedding"):
                with patch("main.is_embedding_model_loaded", return_value=False):
                    with patch("main.LocalLLM") as llm_class:
                        with redirect_stdout(buffer):
                            result = main.handle_command("/model")

        output = buffer.getvalue()
        self.assertEqual(result, "handled")
        self.assertIn("Model durumu", output)
        self.assertIn("phi-4-mini", output)
        self.assertIn("paraphrase-multilingual-Min", output)
        self.assertIn("iLM-L12-v2", output)
        self.assertIn("henüz yüklenmedi", output)
        self.assertIn("yerel Hugging Face snapshot", output)
        self.assertIn("model yüklemez", output)
        llm_class.assert_not_called()

    def test_config_command_shows_active_values(self):
        buffer = io.StringIO()

        with redirect_stdout(buffer):
            result = main.handle_command("/config")

        output = buffer.getvalue()
        self.assertEqual(result, "handled")
        self.assertIn("RAG yapılandırması", output)
        self.assertIn("TOP_K", output)
        self.assertIn("SIMILARITY_THRESHOLD", output)
        self.assertIn("CONTEXT_RELATIVE", output)
        self.assertIn("NEIGHBOR_CHUNK_RADI", output)
        self.assertIn("MAX_CONTEXT_CHUNKS", output)
        self.assertIn("CHUNK_SIZE", output)
        self.assertIn("DOCS_DIR", output)
        self.assertIn("PROJECT_ROOT", output)
        self.assertIn("CLI_HISTORY", output)
        self.assertIn("SESSION_EXPORTS", output)
        self.assertIn("Salt okunur", output)

    def test_stats_command_shows_index_freshness(self):
        buffer = io.StringIO()
        stats = {
            "total_chunks": 11,
            "source_count": 2,
            "db_path": "data/rag.db",
        }

        with patch("main.get_chunk_stats", return_value=stats):
            with patch(
                "main.get_index_freshness",
                return_value=IndexFreshness("current"),
            ):
                with redirect_stdout(buffer):
                    result = main.handle_command("/stats")

        output = buffer.getvalue()
        self.assertEqual(result, "handled")
        self.assertIn("İndeks durumu", output)
        self.assertIn("güncel", output)

    def test_cli_status_is_structured_and_keeps_active_filter(self):
        main._source_filter = "example.txt"

        with patch("main.get_chunk_stats", return_value={"source_count": 3}):
            with patch(
                "main.get_index_freshness",
                return_value=IndexFreshness("current"),
            ):
                status = main.get_cli_status()

        self.assertEqual(status.model_name, main.MODEL_ALIAS)
        self.assertEqual(status.source_count, 3)
        self.assertEqual(status.index_label, "indeks güncel")
        self.assertEqual(status.index_status, "current")
        self.assertEqual(status.source_filter, "example.txt")

    def test_help_lists_model_config_and_document_commands(self):
        buffer = io.StringIO()

        with redirect_stdout(buffer):
            main.print_help()

        output = buffer.getvalue()
        self.assertIn("/model", output)
        self.assertIn("/config", output)
        self.assertIn("/add", output)
        self.assertIn("/remove", output)
        self.assertIn("/benchmark", output)
        self.assertIn("/show", output)
        self.assertIn("/filter", output)
        self.assertIn("/ask", output)

    def test_get_llm_preserves_foundry_output_only_in_debug_mode(self):
        main.DEBUG = True
        fake_llm = object()

        with patch("main.LocalLLM", return_value=fake_llm) as llm_class:
            result = main.get_llm()

        self.assertIs(result, fake_llm)
        llm_class.assert_called_once_with(show_startup_output=True)

    def test_get_llm_suppresses_foundry_output_in_normal_mode(self):
        fake_llm = object()

        with patch("main.LocalLLM", return_value=fake_llm) as llm_class:
            result = main.get_llm()

        self.assertIs(result, fake_llm)
        llm_class.assert_called_once_with(show_startup_output=False)


if __name__ == "__main__":
    unittest.main()
