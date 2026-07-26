import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import main
from app.cli_input import (
    COMMAND_DESCRIPTIONS,
    CLIStatus,
    CLICompleter,
    CLIInputManager,
    PROMPT_TOOLKIT_AVAILABLE,
)

if PROMPT_TOOLKIT_AVAILABLE:
    from app.cli_input import CLIInputLexer
    from prompt_toolkit.completion import Completion
    from prompt_toolkit.document import Document
    from prompt_toolkit.input.defaults import create_pipe_input
    from prompt_toolkit.output import DummyOutput
    from prompt_toolkit.keys import Keys


class FakeReadline:
    __doc__ = "libedit test implementation"

    def __init__(self):
        self.line_buffer = ""
        self.history = []
        self.completer = None
        self.delimiters = None
        self.binding = None
        self.history_length = None

    def get_line_buffer(self):
        return self.line_buffer

    def read_history_file(self, path):
        history_path = Path(path)
        if not history_path.exists():
            raise FileNotFoundError(path)
        self.history = history_path.read_text(encoding="utf-8").splitlines()

    def write_history_file(self, path):
        Path(path).write_text("\n".join(self.history), encoding="utf-8")

    def set_history_length(self, length):
        self.history_length = length

    def set_completer(self, completer):
        self.completer = completer

    def set_completer_delims(self, delimiters):
        self.delimiters = delimiters

    def parse_and_bind(self, binding):
        self.binding = binding


class CLICompleterTests(unittest.TestCase):
    def setUp(self):
        self.readline = FakeReadline()
        self.completer = CLICompleter(
            lambda: [
                {"source_name": "example.txt"},
                {"source_name": "data notes.pdf"},
            ],
            readline_module=self.readline,
            session_id_provider=lambda: [1, 2],
        )

    def test_slash_commands_are_completed(self):
        self.assertEqual(
            self.completer.get_candidates("/so", "/so"),
            ["/sources"],
        )

    def test_source_names_complete_remove_filter_and_ask(self):
        self.assertEqual(
            self.completer.get_candidates("/remove ex", "ex"),
            ["example.txt"],
        )
        self.assertIn(
            "'data notes.pdf'",
            self.completer.get_candidates("/filter d", "d"),
        )
        self.assertEqual(
            self.completer.get_candidates("/ask --source ex", "ex"),
            ["example.txt"],
        )

    def test_add_completes_files_and_directories(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "notes.pdf").touch()
            (root / "notebooks").mkdir()
            line = f"/add {root}/not"
            matches = self.completer.get_candidates(line, f"{root}/not")

        self.assertIn(f"{root}/notes.pdf", matches)
        self.assertIn(f"{root}/notebooks/", matches)

    def test_add_continues_completion_inside_directory_with_spaces(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            spaced_dir = root / "my notes"
            spaced_dir.mkdir()
            (spaced_dir / "document.pdf").touch()

            first_matches = self.completer.get_candidates(
                f"/add {root}/my",
                f"{root}/my",
            )
            escaped_dir = f"{root}/my\\ notes/"
            second_matches = self.completer.get_candidates(
                f"/add {escaped_dir}doc",
                f"{escaped_dir}doc",
            )

        self.assertIn(escaped_dir, first_matches)
        self.assertIn(f"{escaped_dir}document.pdf", second_matches)

    def test_debug_values_are_completed(self):
        self.assertEqual(
            self.completer.get_candidates("/debug o", "o"),
            ["on", "off"],
        )

    def test_session_commands_complete_ids_and_export_formats(self):
        self.assertEqual(
            self.completer.get_candidates("/repeat 2", "2"),
            ["2"],
        )
        self.assertEqual(
            self.completer.get_candidates("/export j", "j"),
            ["json"],
        )

    def test_prompt_candidates_replace_only_active_argument(self):
        candidates, replacement = self.completer.get_prompt_candidates(
            "/ask --source ex"
        )

        self.assertEqual(candidates, ["example.txt"])
        self.assertEqual(replacement, "ex")
        self.assertEqual(
            COMMAND_DESCRIPTIONS["/sources"],
            "İndeksteki kaynakları listeler",
        )


class CLIInputManagerTests(unittest.TestCase):
    def test_history_is_loaded_configured_and_saved_privately(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            history_path = Path(temp_dir) / "data" / "cli_history"
            history_path.parent.mkdir()
            history_path.write_text("RAG nedir?", encoding="utf-8")
            readline = FakeReadline()
            manager = CLIInputManager(
                history_path,
                source_provider=lambda: [],
                readline_module=readline,
                enabled=True,
                history_length=250,
            )

            started = manager.start()
            readline.history.append("/sources")
            saved = manager.save()

            self.assertTrue(started)
            self.assertTrue(saved)
            self.assertEqual(readline.history_length, 250)
            self.assertIs(readline.completer, manager.completer)
            self.assertEqual(readline.delimiters, " \t\n")
            self.assertEqual(readline.binding, "bind ^I rl_complete")
            self.assertEqual(
                history_path.read_text(encoding="utf-8").splitlines(),
                ["RAG nedir?", "/sources"],
            )
            self.assertEqual(history_path.stat().st_mode & 0o777, 0o600)

    def test_non_interactive_input_does_not_touch_history(self):
        readline = FakeReadline()
        manager = CLIInputManager(
            "/missing/history",
            source_provider=lambda: [],
            readline_module=readline,
            enabled=False,
        )

        self.assertFalse(manager.start())
        self.assertFalse(manager.save())
        self.assertIsNone(readline.completer)

    def test_main_starts_and_saves_input_manager(self):
        manager = Mock()
        manager.read.return_value = "/exit"

        with patch("main.CLIInputManager", return_value=manager) as manager_class:
            with patch("main.print_banner"):
                main.main()

        manager_class.assert_called_once_with(
            main.PROJECT_PATHS.history_path,
            source_provider=main.get_indexed_sources,
            fallback_reader=main.read_prompt,
            echo_handler=main.print_submitted_prompt,
            status_provider=main.get_cli_status,
            session_id_provider=main.get_session_ids,
        )
        manager.start.assert_called_once_with()
        manager.read.assert_called_once_with()
        manager.save.assert_called_once_with()

    @unittest.skipUnless(PROMPT_TOOLKIT_AVAILABLE, "prompt-toolkit kurulu değil")
    def test_enhanced_prompt_completes_command_and_persists_history(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            history_path = Path(temp_dir) / "data" / "cli_history"
            status_provider = Mock(
                return_value=CLIStatus(
                    model_name="phi-4-mini",
                    source_count=3,
                    index_label="indeks güncel",
                    index_status="current",
                )
            )
            with create_pipe_input() as pipe_input:
                manager = CLIInputManager(
                    history_path,
                    source_provider=lambda: [],
                    status_provider=status_provider,
                    enabled=True,
                    app_input=pipe_input,
                    app_output=DummyOutput(),
                )
                self.assertTrue(manager.start())
                self.assertTrue(manager.use_enhanced_prompt)

                def send_input():
                    time.sleep(0.05)
                    pipe_input.send_text("/so")
                    time.sleep(0.10)
                    pipe_input.send_text("\t")
                    time.sleep(0.05)
                    pipe_input.send_text("\t\r")

                sender = threading.Thread(target=send_input)
                sender.start()
                result = manager.read()
                sender.join()

                self.assertEqual(result, "/sources")
                status_provider.assert_called_once_with()
                self.assertTrue(manager.save())
                self.assertEqual(
                    history_path.read_text(encoding="utf-8"),
                    "/sources\n",
                )
                self.assertEqual(history_path.stat().st_mode & 0o777, 0o600)

    @unittest.skipUnless(PROMPT_TOOLKIT_AVAILABLE, "prompt-toolkit kurulu değil")
    def test_completion_menu_contains_command_and_description(self):
        completion = Completion(
            "/sources",
            display="/sources",
            display_meta="İndeksteki kaynakları listeler",
        )
        buffer = SimpleNamespace(
            complete_state=SimpleNamespace(
                completions=[completion],
                complete_index=None,
            )
        )

        fragments = CLIInputManager._completion_fragments(buffer)
        rendered_text = "".join(fragment[1] for fragment in fragments)

        self.assertIn("/sources", rendered_text)
        self.assertIn("İndeksteki kaynakları listeler", rendered_text)

    @unittest.skipUnless(PROMPT_TOOLKIT_AVAILABLE, "prompt-toolkit kurulu değil")
    def test_command_lexer_styles_command_and_arguments_separately(self):
        get_line = CLIInputLexer().lex_document(
            Document("/filter example.txt")
        )

        fragments = get_line(0)

        self.assertEqual(fragments[0], ("class:input-command", "/filter"))
        self.assertEqual(
            fragments[1],
            ("class:input-argument", " example.txt"),
        )

    def test_parameter_hint_uses_selected_command_signature(self):
        fragments = CLIInputManager._hint_fragments("/show ")
        rendered_text = "".join(fragment[1] for fragment in fragments)

        self.assertIn("Kullanım", rendered_text)
        self.assertIn("/show <chunk-id>", rendered_text)
        self.assertEqual(CLIInputManager._hint_fragments("RAG nedir?"), [])

    def test_status_line_contains_model_index_source_and_filter(self):
        manager = CLIInputManager(
            "/missing/history",
            source_provider=lambda: [],
            enabled=False,
        )
        manager.current_status = CLIStatus(
            model_name="phi-4-mini",
            source_count=3,
            index_label="indeks güncel",
            index_status="current",
            source_filter="example.txt",
        )

        fragments = manager._status_fragments()
        rendered_text = "".join(fragment[1] for fragment in fragments)

        self.assertIn("phi-4-mini", rendered_text)
        self.assertIn("3 kaynak", rendered_text)
        self.assertIn("indeks güncel", rendered_text)
        self.assertIn("filtre example.txt", rendered_text)

    @unittest.skipUnless(PROMPT_TOOLKIT_AVAILABLE, "prompt-toolkit kurulu değil")
    def test_enhanced_prompt_recalls_existing_history_with_up_arrow(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            history_path = Path(temp_dir) / "data" / "cli_history"
            history_path.parent.mkdir()
            history_path.write_text("RAG nedir?\n", encoding="utf-8")

            with create_pipe_input() as pipe_input:
                manager = CLIInputManager(
                    history_path,
                    source_provider=lambda: [],
                    enabled=True,
                    app_input=pipe_input,
                    app_output=DummyOutput(),
                )
                manager.start()

                def recall_history():
                    time.sleep(0.10)
                    pipe_input.send_text("\x1b[A")
                    time.sleep(0.05)
                    pipe_input.send_text("\r")

                sender = threading.Thread(target=recall_history)
                sender.start()
                result = manager.read()
                sender.join()

        self.assertEqual(result, "RAG nedir?")

    @unittest.skipUnless(PROMPT_TOOLKIT_AVAILABLE, "prompt-toolkit kurulu değil")
    def test_ctrl_c_clears_text_before_it_closes_prompt(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with create_pipe_input() as pipe_input:
                manager = CLIInputManager(
                    Path(temp_dir) / "history",
                    source_provider=lambda: [],
                    enabled=True,
                    app_input=pipe_input,
                    app_output=DummyOutput(),
                )
                manager.start()

                def send_input():
                    time.sleep(0.05)
                    pipe_input.send_text("sil-bunu\x03/exit\r")

                sender = threading.Thread(target=send_input)
                sender.start()
                result = manager.read()
                sender.join()

        self.assertEqual(result, "/exit")

    @unittest.skipUnless(PROMPT_TOOLKIT_AVAILABLE, "prompt-toolkit kurulu değil")
    def test_ctrl_c_on_empty_prompt_raises_keyboard_interrupt(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with create_pipe_input() as pipe_input:
                manager = CLIInputManager(
                    Path(temp_dir) / "history",
                    source_provider=lambda: [],
                    enabled=True,
                    app_input=pipe_input,
                    app_output=DummyOutput(),
                )
                manager.start()
                pipe_input.send_text("\x03")

                with self.assertRaises(KeyboardInterrupt):
                    manager.read()

    @unittest.skipUnless(PROMPT_TOOLKIT_AVAILABLE, "prompt-toolkit kurulu değil")
    def test_ctrl_l_binding_clears_renderer(self):
        buffer = Mock()
        bindings = CLIInputManager._create_key_bindings(buffer)
        binding = next(
            item for item in bindings.bindings
            if item.keys == (Keys.ControlL,)
        )
        event = Mock()

        binding.handler(event)

        event.app.renderer.clear.assert_called_once_with()

    @unittest.skipUnless(PROMPT_TOOLKIT_AVAILABLE, "prompt-toolkit kurulu değil")
    def test_escape_binding_closes_completion_menu(self):
        buffer = Mock()
        bindings = CLIInputManager._create_key_bindings(buffer)
        binding = next(
            item for item in bindings.bindings
            if item.keys == (Keys.Escape,)
        )

        binding.handler(Mock())

        buffer.cancel_completion.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
