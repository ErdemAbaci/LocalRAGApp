import os
import shlex
import sys
from dataclasses import dataclass
from pathlib import Path

try:
    import readline as system_readline
except ImportError:  # pragma: no cover - readline is optional on some platforms
    system_readline = None

try:
    from prompt_toolkit import Application
    from prompt_toolkit.buffer import Buffer
    from prompt_toolkit.completion import Completer, Completion
    from prompt_toolkit.filters import Condition, has_completions
    from prompt_toolkit.formatted_text import fragment_list_to_text, to_formatted_text
    from prompt_toolkit.history import History
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.layout import HSplit, Layout, Window
    from prompt_toolkit.layout.containers import ConditionalContainer
    from prompt_toolkit.layout.controls import BufferControl, FormattedTextControl
    from prompt_toolkit.layout.processors import BeforeInput
    from prompt_toolkit.lexers import Lexer
    from prompt_toolkit.styles import Style
    from prompt_toolkit.widgets import Frame

    PROMPT_TOOLKIT_AVAILABLE = True
except ImportError:  # pragma: no cover - dependency fallback for partial installs
    PROMPT_TOOLKIT_AVAILABLE = False


COMMAND_SPECS = (
    ("/help", "Komut listesini gösterir"),
    ("/stats", "İndeks ve sistem durumunu gösterir"),
    ("/model", "Model ve cache durumunu gösterir"),
    ("/config", "Aktif RAG ayarlarını gösterir"),
    ("/sources", "İndeksteki kaynakları listeler"),
    ("/show", "Bir chunk metnini gösterir"),
    ("/filter", "Aramayı bir kaynakla sınırlar"),
    ("/ask", "Kaynak filtresiyle soru sorar"),
    ("/history", "Bu oturumdaki cevapları listeler"),
    ("/repeat", "Önceki bir soruyu yeniden çalıştırır"),
    ("/export", "Oturumu Markdown veya JSON olarak yazar"),
    ("/doctor", "Sistem sağlığını kontrol eder"),
    ("/add", "TXT veya PDF dokümanı ekler"),
    ("/remove", "Bir dokümanı güvenle siler"),
    ("/benchmark", "Yerel modelleri karşılaştırır"),
    ("/reindex", "Doküman indeksini yeniler"),
    ("/debug", "Teknik çıktıyı açar veya kapatır"),
    ("/exit", "Oturumu kapatır"),
)

INTERACTIVE_COMMANDS = tuple(command for command, _ in COMMAND_SPECS)
COMMAND_DESCRIPTIONS = dict(COMMAND_SPECS)
COMMAND_USAGE = {
    "/show": "/show <chunk-id>",
    "/filter": "/filter <dosya-adı|off>",
    "/ask": "/ask [--source <dosya-adı>] <soru>",
    "/add": "/add <dosya-yolu>",
    "/remove": "/remove <dosya-adı>",
    "/benchmark": "/benchmark [model ...]",
    "/debug": "/debug <on|off>",
    "/repeat": "/repeat [kayıt-id]",
    "/export": "/export <markdown|json> [dosya-yolu]",
}


@dataclass(frozen=True)
class CLIStatus:
    model_name: str
    source_count: int | None
    index_label: str
    index_status: str
    source_filter: str | None = None

PROMPT_STYLE = Style.from_dict(
    {
        "input-frame": "#a9656b",
        "input-title": "bold #c17b80",
        "input-prefix": "bold #c17b80",
        "input-command": "bold #c17b80",
        "input-command-invalid": "bold #b59a68",
        "input-argument": "#b9b1b1",
        "input-hint-label": "#777171",
        "input-hint-value": "#a9656b",
        "status-model": "#b98589",
        "status-value": "#a9a2a2",
        "status-ok": "#78a487",
        "status-warning": "#b59a68",
        "status-separator": "#625d5d",
        "completion-frame": "#6f474b",
        "completion-title": "bold #a9656b",
        "completion-command": "bold #c17b80",
        "completion-meta": "#8f8a8a",
        "completion-selected": "bg:#4a3034 #f0dddd",
        "completion-selected-meta": "bg:#4a3034 #c9b8b8",
    }
)


class CLICompleter:
    def __init__(self, source_provider, readline_module=None, session_id_provider=None):
        self.source_provider = source_provider
        self.session_id_provider = session_id_provider or (lambda: [])
        self.readline = readline_module or system_readline
        self._matches = []

    def __call__(self, text, state):
        if self.readline is None:
            return None

        if state == 0:
            line = self.readline.get_line_buffer()
            self._matches = self.get_candidates(line, text)

        return self._matches[state] if state < len(self._matches) else None

    def get_candidates(self, line, text=None):
        stripped = line.lstrip()

        if not stripped.startswith("/"):
            return []

        if " " not in stripped:
            prefix = text if text is not None else stripped
            return [
                command
                for command in INTERACTIVE_COMMANDS
                if command.startswith(prefix)
            ]

        command, _, argument_text = stripped.partition(" ")
        command = command.casefold()
        current_text = text if text is not None else argument_text.rsplit(" ", 1)[-1]

        if command in {"/remove", "/filter"}:
            candidates = self._source_candidates()
            if command == "/filter":
                candidates.append("off")
            return self._matching_values(candidates, current_text)

        if command == "/ask":
            if argument_text.startswith("--source "):
                source_text = argument_text[len("--source "):]
                if " " not in source_text:
                    return self._matching_values(
                        self._source_candidates(),
                        current_text,
                    )
            elif "--source".startswith(argument_text):
                return ["--source"]
            return []

        if command == "/debug":
            return self._matching_values(["on", "off"], current_text)

        if command == "/repeat":
            return self._matching_values(
                [str(value) for value in self.session_id_provider()],
                current_text,
            )

        if command == "/export":
            format_text, separator, path_text = argument_text.partition(" ")
            if not separator:
                return self._matching_values(
                    ["markdown", "json"],
                    current_text,
                )
            return self._path_candidates(path_text)

        if command == "/add":
            return self._path_candidates(argument_text)

        return []

    def get_prompt_candidates(self, line):
        stripped = line.lstrip()
        if not stripped.startswith("/"):
            return [], ""

        if " " not in stripped:
            return self.get_candidates(line, stripped), stripped

        command, _, argument_text = stripped.partition(" ")
        if command.casefold() == "/add":
            replacement = argument_text
        else:
            replacement = argument_text.rsplit(" ", 1)[-1]

        return self.get_candidates(line, replacement), replacement

    def _source_candidates(self):
        try:
            sources = self.source_provider()
        except Exception:
            return []

        names = []
        for source in sources:
            name = source.get("source_name") if isinstance(source, dict) else str(source)
            if name and name not in names:
                names.append(name)
        return names

    @staticmethod
    def _matching_values(values, prefix):
        clean_prefix = prefix.strip("'\"").casefold()
        return [
            shlex.quote(value)
            for value in values
            if value.casefold().startswith(clean_prefix)
        ]

    @staticmethod
    def _path_candidates(argument_text):
        raw_value = argument_text.strip()
        if raw_value.startswith(("'", '"')):
            quote = raw_value[0]
            raw_value = raw_value[1:-1] if raw_value.endswith(quote) else raw_value[1:]
        raw_value = raw_value.replace("\\ ", " ")

        expanded_value = os.path.expanduser(raw_value or ".")
        candidate_path = Path(expanded_value)

        if raw_value.endswith(os.sep):
            parent = candidate_path
            prefix = ""
            display_parent = raw_value
        else:
            parent = candidate_path.parent
            prefix = candidate_path.name
            display_parent = str(Path(raw_value).parent)
            if display_parent == ".":
                display_parent = ""

        try:
            entries = sorted(parent.iterdir(), key=lambda path: path.name.casefold())
        except OSError:
            return []

        matches = []
        for entry in entries:
            if not entry.name.casefold().startswith(prefix.casefold()):
                continue

            display_value = (
                str(Path(display_parent) / entry.name)
                if display_parent
                else entry.name
            )
            if entry.is_dir():
                display_value += os.sep
            matches.append(
                display_value.replace("\\", "\\\\").replace(" ", "\\ ")
            )

        return matches


if PROMPT_TOOLKIT_AVAILABLE:
    class CLIInputLexer(Lexer):
        def lex_document(self, document):
            lines = document.lines

            def get_line(line_number):
                if line_number >= len(lines):
                    return []

                line = lines[line_number]
                stripped = line.lstrip()
                leading_space_count = len(line) - len(stripped)

                if not stripped.startswith("/"):
                    return [("", line)]

                command, separator, arguments = stripped.partition(" ")
                is_known_or_partial = any(
                    candidate.startswith(command.casefold())
                    for candidate in INTERACTIVE_COMMANDS
                )
                command_style = (
                    "class:input-command"
                    if is_known_or_partial
                    else "class:input-command-invalid"
                )
                fragments = []
                if leading_space_count:
                    fragments.append(("", line[:leading_space_count]))
                fragments.append((command_style, command))
                if separator:
                    fragments.append(("class:input-argument", f" {arguments}"))
                return fragments

            return get_line


    class PromptToolkitCompleter(Completer):
        def __init__(self, cli_completer):
            self.cli_completer = cli_completer

        def get_completions(self, document, complete_event):
            line = document.text_before_cursor
            candidates, replacement = self.cli_completer.get_prompt_candidates(line)
            command = line.lstrip().partition(" ")[0]

            for candidate in candidates:
                description = COMMAND_DESCRIPTIONS.get(candidate)
                if description is None:
                    if command in {"/remove", "/filter", "/ask"}:
                        description = "İndekslenmiş kaynak"
                    elif command == "/add":
                        description = "Dosya veya klasör"
                    elif command == "/debug":
                        description = "Debug seçeneği"
                    elif command == "/repeat":
                        description = "Oturum kaydı"
                    elif command == "/export":
                        description = "Dışa aktarım seçeneği"
                    else:
                        description = ""

                yield Completion(
                    candidate,
                    start_position=-len(replacement),
                    display=candidate,
                    display_meta=description,
                )


    class PlainFileHistory(History):
        def __init__(self, history_path, history_length=500):
            self.history_path = Path(history_path)
            self.history_length = history_length
            super().__init__()

        def load_history_strings(self):
            try:
                entries = self.history_path.read_text(encoding="utf-8").splitlines()
            except (FileNotFoundError, OSError):
                return []

            entries = [entry for entry in entries if entry.strip()]
            return reversed(entries[-self.history_length:])

        def store_string(self, string):
            clean_string = " ".join(string.splitlines()).strip()
            if not clean_string:
                return

            try:
                self.history_path.parent.mkdir(parents=True, exist_ok=True)
                with self.history_path.open("a", encoding="utf-8") as history_file:
                    history_file.write(f"{clean_string}\n")
                self.history_path.chmod(0o600)
            except OSError:
                return


class CLIInputManager:
    def __init__(
        self,
        history_path,
        source_provider,
        readline_module=None,
        enabled=None,
        history_length=500,
        fallback_reader=None,
        echo_handler=None,
        status_provider=None,
        session_id_provider=None,
        app_input=None,
        app_output=None,
    ):
        self.history_path = Path(history_path)
        self.readline = readline_module or system_readline
        self.history_length = history_length
        interactive_terminal = sys.stdin.isatty() and sys.stdout.isatty()
        self.enabled = interactive_terminal if enabled is None else bool(enabled)
        self.use_enhanced_prompt = (
            self.enabled
            and PROMPT_TOOLKIT_AVAILABLE
            and readline_module is None
        )
        self.fallback_reader = fallback_reader or (lambda: input("> "))
        self.echo_handler = echo_handler
        self.status_provider = status_provider
        self.current_status = None
        self.app_input = app_input
        self.app_output = app_output
        self.completer = CLICompleter(
            source_provider,
            readline_module=self.readline,
            session_id_provider=session_id_provider,
        )
        self.prompt_completer = (
            PromptToolkitCompleter(self.completer)
            if self.use_enhanced_prompt
            else None
        )
        self.history = (
            PlainFileHistory(self.history_path, history_length)
            if self.use_enhanced_prompt
            else None
        )

    def start(self):
        if not self.enabled:
            return False

        if self.use_enhanced_prompt:
            try:
                self.history_path.parent.mkdir(parents=True, exist_ok=True)
                if not self.history_path.exists():
                    self.history_path.touch(mode=0o600)
                self.history_path.chmod(0o600)
            except OSError:
                return False
            return True

        if self.readline is None:
            return False

        try:
            self.readline.read_history_file(str(self.history_path))
        except (FileNotFoundError, OSError):
            pass

        self.readline.set_history_length(self.history_length)
        self.readline.set_completer(self.completer)
        self.readline.set_completer_delims(" \t\n")

        if "libedit" in (self.readline.__doc__ or "").casefold():
            self.readline.parse_and_bind("bind ^I rl_complete")
        else:
            self.readline.parse_and_bind("tab: complete")

        return True

    def read(self):
        if not self.use_enhanced_prompt:
            return self.fallback_reader()

        self.current_status = self._load_status()
        result = self._create_application().run()
        if self.echo_handler is not None and result:
            self.echo_handler(result)
        return result

    def save(self):
        if not self.enabled:
            return False

        if self.use_enhanced_prompt:
            return self._trim_enhanced_history()

        if self.readline is None:
            return False

        try:
            self.history_path.parent.mkdir(parents=True, exist_ok=True)
            if not self.history_path.exists():
                self.history_path.touch(mode=0o600)
            self.readline.write_history_file(str(self.history_path))
            self.history_path.chmod(0o600)
        except OSError:
            return False

        return True

    def _create_application(self):
        def accept_input(buffer):
            application.exit(result=buffer.text)

        buffer = Buffer(
            completer=self.prompt_completer,
            complete_while_typing=True,
            history=self.history,
            accept_handler=accept_input,
            multiline=False,
        )

        input_control = BufferControl(
            buffer=buffer,
            lexer=CLIInputLexer(),
            input_processors=[
                BeforeInput([("class:input-prefix", "› ")]),
            ],
        )
        input_window = Window(
            input_control,
            height=1,
            wrap_lines=False,
        )
        hint_window = Window(
            FormattedTextControl(lambda: self._hint_fragments(buffer.text)),
            height=1,
            dont_extend_height=True,
        )
        hint_container = ConditionalContainer(
            hint_window,
            filter=Condition(lambda: self._parameter_hint(buffer.text) is not None),
        )
        input_frame = Frame(
            HSplit([input_window, hint_container]),
            title=[("class:input-title", " Local RAG ")],
            style="class:input-frame",
        )

        status_window = Window(
            FormattedTextControl(self._status_fragments),
            height=1,
            dont_extend_height=True,
        )
        status_container = ConditionalContainer(
            status_window,
            filter=Condition(lambda: self.current_status is not None),
        )
        input_group = HSplit([input_frame, status_container])

        completion_control = FormattedTextControl(
            lambda: self._completion_fragments(buffer),
        )
        completion_window = Window(
            completion_control,
            height=lambda: self._completion_height(buffer),
            dont_extend_height=True,
        )
        completion_frame = Frame(
            completion_window,
            title=[("class:completion-title", " Öneriler ")],
            style="class:completion-frame",
        )
        completion_container = ConditionalContainer(
            completion_frame,
            filter=has_completions,
        )

        key_bindings = self._create_key_bindings(buffer)
        root = HSplit(
            [completion_container, input_group],
            padding=1,
            padding_char=" ",
        )
        application = Application(
            layout=Layout(root, focused_element=input_control),
            key_bindings=key_bindings,
            style=PROMPT_STYLE,
            full_screen=False,
            mouse_support=False,
            erase_when_done=True,
            input=self.app_input,
            output=self.app_output,
        )
        return application

    def _load_status(self):
        if self.status_provider is None:
            return None

        try:
            return self.status_provider()
        except Exception:
            return None

    @staticmethod
    def _parameter_hint(value):
        command = value.lstrip().partition(" ")[0].casefold()
        if command in COMMAND_USAGE:
            return COMMAND_USAGE[command]

        matches = [
            usage
            for candidate, usage in COMMAND_USAGE.items()
            if command and candidate.startswith(command)
        ]
        return matches[0] if len(matches) == 1 else None

    @classmethod
    def _hint_fragments(cls, value):
        hint = cls._parameter_hint(value)
        if hint is None:
            return []
        return [
            ("class:input-hint-label", "  Kullanım  "),
            ("class:input-hint-value", hint),
        ]

    def _status_fragments(self):
        status = self.current_status
        if status is None:
            return []

        source_count = "? kaynak" if status.source_count is None else (
            f"{status.source_count} kaynak"
        )
        index_style = (
            "class:status-ok"
            if status.index_status == "current"
            else "class:status-warning"
        )
        source_filter = status.source_filter or "kapalı"
        if len(source_filter) > 24:
            source_filter = f"{source_filter[:21]}..."

        return [
            ("class:status-model", f" {status.model_name}"),
            ("class:status-separator", "  ·  "),
            ("class:status-value", source_count),
            ("class:status-separator", "  ·  "),
            (index_style, status.index_label),
            ("class:status-separator", "  ·  "),
            ("class:status-value", f"filtre {source_filter}"),
        ]

    @staticmethod
    def _create_key_bindings(buffer):
        bindings = KeyBindings()

        @bindings.add("enter")
        def accept(event):
            buffer.validate_and_handle()

        @bindings.add("tab")
        def complete(event):
            state = buffer.complete_state
            if state is None:
                buffer.start_completion(select_first=True)
            elif state.current_completion is None:
                buffer.complete_next()
            else:
                buffer.apply_completion(state.current_completion)

        @bindings.add("down")
        def move_down(event):
            if buffer.complete_state is not None:
                buffer.complete_next()
            else:
                buffer.history_forward()

        @bindings.add("up")
        def move_up(event):
            if buffer.complete_state is not None:
                buffer.complete_previous()
            else:
                buffer.history_backward()

        @bindings.add("escape")
        def cancel_completion(event):
            buffer.cancel_completion()

        @bindings.add("c-c")
        def interrupt(event):
            if buffer.text:
                buffer.reset()
            else:
                event.app.exit(exception=KeyboardInterrupt)

        @bindings.add("c-l")
        def clear_screen(event):
            event.app.renderer.clear()

        @bindings.add("c-d")
        def end_of_input(event):
            if not buffer.text:
                event.app.exit(exception=EOFError)
            else:
                buffer.delete()

        return bindings

    @staticmethod
    def _completion_height(buffer):
        state = buffer.complete_state
        return min(len(state.completions), 6) if state is not None else 1

    @staticmethod
    def _completion_fragments(buffer):
        state = buffer.complete_state
        if state is None or not state.completions:
            return []

        completions = state.completions
        selected_index = state.complete_index
        visible_count = min(len(completions), 6)
        effective_index = selected_index if selected_index is not None else 0
        start = max(0, effective_index - visible_count + 1)
        start = min(start, max(0, len(completions) - visible_count))
        fragments = []

        for index, completion in enumerate(completions[start:start + visible_count], start):
            selected = selected_index is not None and index == effective_index
            command_style = (
                "class:completion-selected"
                if selected
                else "class:completion-command"
            )
            meta_style = (
                "class:completion-selected-meta"
                if selected
                else "class:completion-meta"
            )
            command = fragment_list_to_text(to_formatted_text(completion.display))
            meta = fragment_list_to_text(to_formatted_text(completion.display_meta))
            fragments.append((command_style, f" {command:<16}"))
            fragments.append((meta_style, f" {meta}"))
            if index < start + visible_count - 1:
                fragments.append(("", "\n"))

        return fragments

    def _trim_enhanced_history(self):
        try:
            entries = self.history_path.read_text(encoding="utf-8").splitlines()
            entries = [entry for entry in entries if entry.strip()]
            entries = entries[-self.history_length:]
            suffix = "\n" if entries else ""
            self.history_path.write_text(
                "\n".join(entries) + suffix,
                encoding="utf-8",
            )
            self.history_path.chmod(0o600)
        except OSError:
            return False

        return True
