import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import main
from app import database
from app.health import check_documents, check_foundry, check_index, run_health_checks
from app.llm import MODEL_ALIAS


def create_foundry_cache(root, include_model=True):
    foundry_home = root / ".foundry"
    cache_path = foundry_home / "cache" / "models"
    cache_path.mkdir(parents=True)
    config = {"serviceSettings": {"cacheDirectoryPath": str(cache_path)}}
    (foundry_home / "foundry.config.json").write_text(
        json.dumps(config),
        encoding="utf-8",
    )

    if include_model:
        model_path = cache_path / "Microsoft" / "Phi-4-mini-instruct-generic-gpu-5" / "v5"
        model_path.mkdir(parents=True)
        (model_path / "inference_model.json").write_text("{}", encoding="utf-8")
        (model_path / "model.onnx.data").write_bytes(b"model")

    return foundry_home


class HealthCheckTests(unittest.TestCase):
    def test_healthy_system_passes_all_checks(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            docs_dir = root / "docs"
            docs_dir.mkdir()
            (docs_dir / "notes.txt").write_text("RAG notları", encoding="utf-8")
            db_path = root / "rag.db"
            foundry_home = create_foundry_cache(root)

            with patch.object(database, "DB_PATH", db_path):
                database.init_db()
                database.insert_chunk(
                    "notes.txt",
                    "RAG notları",
                    [0.1] * 384,
                    source_type="txt",
                    chunk_index=1,
                )
                checks = run_health_checks(
                    docs_dir=docs_dir,
                    db_path=db_path,
                    foundry_home=foundry_home,
                    executable_finder=lambda name: "/usr/local/bin/foundry",
                )

        self.assertTrue(all(check.status == "ok" for check in checks))

    def test_missing_documents_and_database_return_actionable_warnings(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            foundry_home = create_foundry_cache(root)
            checks = run_health_checks(
                docs_dir=root / "missing-docs",
                db_path=root / "missing.db",
                foundry_home=foundry_home,
                executable_finder=lambda name: "/usr/local/bin/foundry",
            )

        by_name = {check.name: check for check in checks}
        self.assertEqual(by_name["Dokümanlar"].status, "error")
        self.assertEqual(by_name["Veritabanı"].status, "warning")
        self.assertEqual(by_name["Embedding indeksi"].status, "warning")
        self.assertIn("/reindex", by_name["Veritabanı"].solution)

    def test_invalid_embedding_is_reported(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "rag.db"

            with patch.object(database, "DB_PATH", db_path):
                database.init_db()
                database.insert_chunk("broken.txt", "bozuk", [0.1, 0.2])
                checks = check_index(db_path=db_path)

        embedding_check = next(check for check in checks if check.name == "Embedding indeksi")
        self.assertEqual(embedding_check.status, "error")
        self.assertIn("/reindex", embedding_check.solution)

    def test_foundry_errors_and_missing_model_are_reported(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            missing_model_home = create_foundry_cache(root, include_model=False)
            missing_config_home = root / "missing-foundry"
            finder = lambda name: "/usr/local/bin/foundry"
            failed_checks = check_foundry(
                foundry_home=missing_config_home,
                executable_finder=finder,
            )
            missing_model_checks = check_foundry(
                foundry_home=missing_model_home,
                executable_finder=finder,
            )

        self.assertEqual(failed_checks[0].status, "error")
        self.assertEqual(failed_checks[1].status, "warning")
        self.assertEqual(missing_model_checks[0].status, "ok")
        self.assertEqual(missing_model_checks[1].status, "error")
        self.assertIn("foundry model download", missing_model_checks[1].solution)

    def test_missing_foundry_command_is_reported(self):
        checks = check_foundry(
            executable_finder=lambda name: None,
        )

        self.assertEqual(checks[0].status, "error")
        self.assertEqual(checks[1].status, "warning")

    def test_doctor_command_prints_summary(self):
        fake_checks = [
            SimpleNamespace(name="Dokümanlar", status="ok", message="hazır", solution=None),
            SimpleNamespace(
                name="Embedding indeksi",
                status="warning",
                message="boş",
                solution="/reindex çalıştır.",
            ),
        ]
        buffer = io.StringIO()

        with patch("main.run_health_checks", return_value=fake_checks):
            with redirect_stdout(buffer):
                result = main.handle_command("/doctor")

        output = buffer.getvalue()
        self.assertEqual(result, "handled")
        self.assertIn("OK", output)
        self.assertIn("Dokümanlar", output)
        self.assertIn("hazır", output)
        self.assertIn("UYARI", output)
        self.assertIn("Embedding indeksi", output)
        self.assertIn("boş", output)
        self.assertIn("Çözüm: /reindex çalıştır.", output)
        self.assertIn("1 başarılı", output)
        self.assertIn("1 uyarı", output)
        self.assertIn("0 hata", output)


if __name__ == "__main__":
    unittest.main()
