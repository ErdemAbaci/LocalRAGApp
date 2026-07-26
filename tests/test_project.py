import io
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import main
from app import database, health, ingest
from app.project import (
    DEFAULT_PROJECT_ROOT,
    PROJECT_ENV_VAR,
    ProjectConfigurationError,
    get_project_paths,
    resolve_project_root,
)


class ProjectPathTests(unittest.TestCase):
    def tearDown(self):
        main.configure_project(DEFAULT_PROJECT_ROOT, environ={})

    def test_default_project_is_independent_from_current_directory(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch("pathlib.Path.cwd", return_value=Path(temp_dir)):
                root = resolve_project_root(environ={})

        self.assertEqual(root, DEFAULT_PROJECT_ROOT.resolve())

    def test_explicit_path_overrides_environment(self):
        with tempfile.TemporaryDirectory() as explicit_dir:
            with tempfile.TemporaryDirectory() as environment_dir:
                root = resolve_project_root(
                    explicit_dir,
                    environ={PROJECT_ENV_VAR: environment_dir},
                )

        self.assertEqual(root, Path(explicit_dir).resolve())

    def test_environment_selects_project_and_paths(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            paths = get_project_paths(
                environ={PROJECT_ENV_VAR: temp_dir},
            )

        root = Path(temp_dir).resolve()
        self.assertEqual(paths.docs_dir, root / "docs")
        self.assertEqual(paths.db_path, root / "data" / "rag.db")
        self.assertEqual(paths.history_path, root / "data" / "cli_history")
        self.assertEqual(paths.session_export_dir, root / "data" / "exports")

    def test_invalid_project_path_is_rejected(self):
        with self.assertRaises(ProjectConfigurationError):
            resolve_project_root("/definitely/missing/local-rag-project", environ={})

    def test_configure_project_updates_all_runtime_modules(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            paths = main.configure_project(temp_dir, environ={})

            self.assertEqual(main.DOCS_DIR, paths.docs_dir)
            self.assertEqual(main.DB_PATH, paths.db_path)
            self.assertEqual(database.DB_PATH, paths.db_path)
            self.assertEqual(ingest.DOCS_DIR, paths.docs_dir)
            self.assertEqual(health.DOCS_DIR, paths.docs_dir)

    def test_cli_project_option_is_visible_in_config(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            buffer = io.StringIO()

            with redirect_stdout(buffer):
                exit_code = main.cli(["--project", temp_dir, "config"])

        self.assertEqual(exit_code, 0)
        self.assertEqual(main.PROJECT_PATHS.root, Path(temp_dir).resolve())
        self.assertIn("PROJECT_ROOT", buffer.getvalue())

    def test_cli_reads_local_rag_home_environment_variable(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch.dict(os.environ, {PROJECT_ENV_VAR: temp_dir}):
                with patch("main.execute_command", return_value=True):
                    exit_code = main.cli(["stats"])

            self.assertEqual(exit_code, 0)
            self.assertEqual(main.PROJECT_PATHS.root, Path(temp_dir).resolve())


if __name__ == "__main__":
    unittest.main()
