import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import main
from app.rag_service import RAGResult, RAGSource, RAGTimings
from app.session import SessionExportError, SessionHistory


FIXED_TIME = datetime(2026, 7, 17, 21, 30, tzinfo=timezone.utc)


def make_result(question="RAG nedir?", source_filter=None):
    return RAGResult(
        question=question,
        answer="RAG, retrieval ve generation adımlarını birleştirir.",
        mode="generative",
        best_score=0.6123,
        sources=(RAGSource(
            id=7,
            source_name="example.txt",
            chunk_text="Dışa aktarılmaması gereken tam chunk metni.",
            score=0.6123,
            source_type="txt",
            chunk_index=1,
        ),),
        timings=RAGTimings(0.1, 1.2, 1.3),
        source_filter=source_filter,
    )


class SessionHistoryTests(unittest.TestCase):
    def setUp(self):
        self.history = SessionHistory(now_factory=lambda: FIXED_TIME)

    def test_entries_are_numbered_and_can_be_selected(self):
        first = self.history.add_result(make_result())
        second = self.history.add_result(make_result("Embedding nedir?"))

        self.assertEqual(self.history.ids(), [1, 2])
        self.assertEqual(self.history.get().id, 2)
        self.assertEqual(self.history.get(1), first)
        self.assertEqual(second.question, "Embedding nedir?")
        self.assertNotIn("chunk_text", second.sources[0])

    def test_markdown_and_json_exports_are_structured(self):
        self.history.add_result(make_result(source_filter="example.txt"))

        with tempfile.TemporaryDirectory() as temp_dir:
            export_dir = Path(temp_dir) / "exports"
            markdown_path = self.history.export("markdown", export_dir)
            json_path = self.history.export("json", export_dir)
            markdown = markdown_path.read_text(encoding="utf-8")
            payload = json.loads(json_path.read_text(encoding="utf-8"))

        self.assertEqual(markdown_path.name, "session-20260717-213000.md")
        self.assertIn("## 1. Soru", markdown)
        self.assertIn("RAG nedir?", markdown)
        self.assertNotIn("Dışa aktarılmaması gereken", markdown)
        self.assertEqual(payload["entries"][0]["source_filter"], "example.txt")
        self.assertEqual(payload["entries"][0]["sources"][0]["id"], 7)
        self.assertNotIn("chunk_text", payload["entries"][0]["sources"][0])

    def test_export_rejects_empty_history_bad_format_and_overwrite(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            export_dir = Path(temp_dir)
            with self.assertRaisesRegex(SessionExportError, "kaydı bulunmuyor"):
                self.history.export("json", export_dir)

            self.history.add_result(make_result())
            with self.assertRaisesRegex(SessionExportError, "markdown veya json"):
                self.history.export("xml", export_dir)

            destination = export_dir / "session.json"
            destination.write_text("koru", encoding="utf-8")
            with self.assertRaisesRegex(SessionExportError, "üzerine yazılmadı"):
                self.history.export("json", export_dir, destination)
            self.assertEqual(destination.read_text(encoding="utf-8"), "koru")


class SessionCommandTests(unittest.TestCase):
    def setUp(self):
        main._session_history.clear()

    def tearDown(self):
        main._session_history.clear()
        main._source_filter = None

    def test_history_lists_entries_and_repeat_preserves_original_filter(self):
        main._session_history.add_result(
            make_result(source_filter="example.txt")
        )
        output = io.StringIO()

        with redirect_stdout(output):
            history_result = main.handle_command("/history")
        with patch("main.answer_question", return_value=True) as answer:
            with redirect_stdout(output):
                repeat_result = main.handle_command("/repeat 1")

        self.assertEqual(history_result, "handled")
        self.assertEqual(repeat_result, "handled")
        self.assertIn("Oturum geçmişi", output.getvalue())
        self.assertIn("RAG nedir?", output.getvalue())
        answer.assert_called_once_with(
            "RAG nedir?",
            source_name="example.txt",
            use_active_filter=False,
        )

    def test_export_command_writes_json_to_project_export_directory(self):
        main._session_history.add_result(make_result())

        with tempfile.TemporaryDirectory() as temp_dir:
            paths = SimpleNamespace(session_export_dir=Path(temp_dir) / "exports")
            output = io.StringIO()
            with patch.object(main, "PROJECT_PATHS", paths):
                with redirect_stdout(output):
                    result = main.handle_command("/export json")

            exported_files = list(paths.session_export_dir.glob("*.json"))

        self.assertEqual(result, "handled")
        self.assertEqual(len(exported_files), 1)
        self.assertIn("Oturum dışa aktarıldı", output.getvalue())

    def test_cancelled_answer_is_not_added_to_history(self):
        service = MagicMock()
        service.answer.side_effect = KeyboardInterrupt
        output = io.StringIO()

        with patch("main.warn_if_index_is_stale"):
            with patch("main.RAGService", return_value=service):
                with redirect_stdout(output):
                    success = main.answer_question("İptal edilecek soru")

        self.assertFalse(success)
        self.assertEqual(main._session_history.entries, [])
        self.assertIn("kısmi cevap kaydedilmedi", output.getvalue())

    def test_successful_answer_is_added_after_rendering(self):
        service = MagicMock()
        service.answer.return_value = make_result()

        with patch("main.warn_if_index_is_stale"):
            with patch("main.RAGService", return_value=service):
                with patch("main.print_answer"):
                    with patch("main.print_sources"):
                        with patch("main.print_performance"):
                            success = main.answer_question("RAG nedir?")

        self.assertTrue(success)
        self.assertEqual(len(main._session_history.entries), 1)
        self.assertEqual(main._session_history.entries[0].question, "RAG nedir?")


if __name__ == "__main__":
    unittest.main()
