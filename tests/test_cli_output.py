import io
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

import main
from app.cli_output import (
    ANSWER_MODE_STYLES,
    print_answer,
    print_issue,
    print_performance,
)


class CliOutputTests(unittest.TestCase):
    def tearDown(self):
        main.DEBUG = False

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

    def test_answer_modes_have_distinct_turkish_labels_and_styles(self):
        expected_modes = {
            "generative": ("Üretken", "cyan"),
            "extractive": ("Doğrudan", "green"),
            "fallback_extractive": ("Kaynak metni", "yellow"),
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


if __name__ == "__main__":
    unittest.main()
