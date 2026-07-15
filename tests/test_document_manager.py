import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import main
from pypdf import PdfWriter
from app import database
from app.document_manager import (
    DocumentManagementError,
    add_document,
    remove_document,
    resolve_managed_document,
)
from app.index_state import build_source_manifest, get_index_freshness


def sample_chunk(source_name):
    return {
        "source_name": source_name,
        "source_type": "txt",
        "page_number": None,
        "chunk_index": 0,
        "chunk_text": "RAG notları",
        "embedding": [0.1] * 384,
    }


class DocumentManagerTests(unittest.TestCase):
    def test_add_document_copies_valid_txt_without_overwriting(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "RAG Notları.TXT"
            source.write_text("RAG dokümanlardan bilgi getirir.", encoding="utf-8")
            docs_dir = root / "docs"

            destination = add_document(source, docs_dir)

            self.assertEqual(destination, docs_dir / source.name)
            self.assertEqual(destination.read_text(encoding="utf-8"), source.read_text())

            destination.write_text("korunan içerik", encoding="utf-8")
            with self.assertRaisesRegex(DocumentManagementError, "üzerine yazılmadı"):
                add_document(source, docs_dir)

            self.assertEqual(destination.read_text(encoding="utf-8"), "korunan içerik")

    def test_add_rejects_unsupported_empty_and_non_utf8_files(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            docs_dir = root / "docs"
            unsupported = root / "notes.md"
            unsupported.write_text("metin", encoding="utf-8")
            empty = root / "empty.txt"
            empty.write_text("  \n", encoding="utf-8")
            invalid = root / "invalid.txt"
            invalid.write_bytes(b"\xff\xfe")

            with self.assertRaisesRegex(DocumentManagementError, "TXT ve PDF"):
                add_document(unsupported, docs_dir)

            with self.assertRaisesRegex(DocumentManagementError, "indekslenebilir metin"):
                add_document(empty, docs_dir)

            with self.assertRaisesRegex(DocumentManagementError, "UTF-8"):
                add_document(invalid, docs_dir)

    def test_add_accepts_text_pdf_and_rejects_empty_pdf(self):
        project_root = Path(__file__).resolve().parents[1]
        text_pdf = project_root / "docs" / "datamining.pdf"

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            docs_dir = root / "docs"
            destination = add_document(text_pdf, docs_dir)
            empty_pdf = root / "empty.pdf"
            writer = PdfWriter()
            writer.add_blank_page(width=100, height=100)

            with empty_pdf.open("wb") as file_handle:
                writer.write(file_handle)

            with self.assertRaisesRegex(DocumentManagementError, "indekslenebilir metin"):
                add_document(empty_pdf, docs_dir)

        self.assertEqual(destination.name, "datamining.pdf")

    def test_failed_copy_validation_removes_partial_destination(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "notes.txt"
            source.write_text("geçerli içerik", encoding="utf-8")
            docs_dir = root / "docs"

            with patch(
                "app.document_manager.validate_document",
                side_effect=[source, DocumentManagementError("kopya bozuk")],
            ):
                with self.assertRaisesRegex(DocumentManagementError, "kopya bozuk"):
                    add_document(source, docs_dir)

            self.assertFalse((docs_dir / "notes.txt").exists())

    def test_remove_accepts_only_a_managed_file_name(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            docs_dir = root / "docs"
            docs_dir.mkdir()
            managed = docs_dir / "notes.txt"
            managed.write_text("RAG notları", encoding="utf-8")
            outside = root / "outside.txt"
            outside.write_text("korunmalı", encoding="utf-8")

            with self.assertRaisesRegex(DocumentManagementError, "yalnızca adını"):
                resolve_managed_document("../outside.txt", docs_dir)

            self.assertTrue(outside.exists())
            removed = remove_document("notes.txt", docs_dir)
            self.assertEqual(removed, managed)
            self.assertFalse(managed.exists())

    def test_add_and_remove_are_reported_by_index_freshness(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            docs_dir = root / "docs"
            docs_dir.mkdir()
            indexed = docs_dir / "indexed.txt"
            indexed.write_text("İndekslenmiş RAG notları", encoding="utf-8")
            external = root / "new.txt"
            external.write_text("Yeni embedding notları", encoding="utf-8")
            db_path = root / "rag.db"

            with patch.object(database, "DB_PATH", db_path):
                database.init_db()
                database.replace_chunks(
                    [sample_chunk("indexed.txt")],
                    source_manifest=build_source_manifest(docs_dir),
                )
                add_document(external, docs_dir)
                added_state = get_index_freshness(docs_dir, db_path)
                remove_document("indexed.txt", docs_dir)
                changed_state = get_index_freshness(docs_dir, db_path)

        self.assertEqual(added_state.added, ("new.txt",))
        self.assertEqual(changed_state.added, ("new.txt",))
        self.assertEqual(changed_state.deleted, ("indexed.txt",))


class DocumentCommandTests(unittest.TestCase):
    def tearDown(self):
        main.DEBUG = False

    def test_interactive_add_preserves_case_and_spaces_in_path(self):
        with patch("main.add_document_command", return_value=True) as add_command:
            result = main.handle_command('/add "/Tmp/RAG Notes.TXT"')

        self.assertEqual(result, "handled")
        add_command.assert_called_once_with("/Tmp/RAG Notes.TXT")

    def test_remove_requires_confirmation_and_can_be_cancelled(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            docs_dir = Path(temp_dir) / "docs"
            docs_dir.mkdir()
            document = docs_dir / "notes.txt"
            document.write_text("RAG notları", encoding="utf-8")
            buffer = io.StringIO()

            with patch.object(main, "DOCS_DIR", docs_dir):
                with patch("main.console.input", return_value="hayır"):
                    with redirect_stdout(buffer):
                        success = main.remove_document_command("notes.txt")

            self.assertTrue(success)
            self.assertTrue(document.exists())
            self.assertIn("iptal edildi", buffer.getvalue())

    def test_remove_with_confirmation_deletes_document(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            docs_dir = Path(temp_dir) / "docs"
            docs_dir.mkdir()
            document = docs_dir / "notes.txt"
            document.write_text("RAG notları", encoding="utf-8")
            buffer = io.StringIO()

            with patch.object(main, "DOCS_DIR", docs_dir):
                with patch("main.console.input", return_value="evet"):
                    with redirect_stdout(buffer):
                        success = main.remove_document_command("notes.txt")

            self.assertTrue(success)
            self.assertFalse(document.exists())

    def test_cli_add_and_remove_forward_arguments(self):
        with patch("main.add_document_command", return_value=True) as add_command:
            add_exit_code = main.cli(["add", "/Tmp/RAG Notes.TXT"])

        with patch("main.remove_document_command", return_value=True) as remove_command:
            remove_exit_code = main.cli(["remove", "notes.txt", "--yes"])

        self.assertEqual(add_exit_code, 0)
        self.assertEqual(remove_exit_code, 0)
        add_command.assert_called_once_with("/Tmp/RAG Notes.TXT")
        remove_command.assert_called_once_with("notes.txt", True)

    def test_add_and_remove_usage_errors_are_actionable(self):
        buffer = io.StringIO()

        with redirect_stdout(buffer):
            add_result = main.handle_command("/add")
            remove_result = main.handle_command("/remove too many arguments")

        output = buffer.getvalue()
        self.assertEqual(add_result, "handled")
        self.assertEqual(remove_result, "handled")
        self.assertIn("Kullanım: /add <dosya-yolu>", output)
        self.assertIn("Kullanım: /remove <dosya-adı>", output)


if __name__ == "__main__":
    unittest.main()
