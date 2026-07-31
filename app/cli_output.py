from contextlib import contextmanager, nullcontext
import sys

from rich import box
from rich.console import Console
from rich.live import Live
from rich.padding import Padding
from rich.panel import Panel
from rich.table import Table
from rich.text import Text


console = Console(highlight=False)

PRIMARY = "#a9656b"
PRIMARY_BRIGHT = "#c17b80"
PRIMARY_DARK = "#6f474b"
MUTED_GREEN = "#78a487"
MUTED_AMBER = "#b59a68"
DEBUG_COLOR = "#9b708e"

ISSUE_STYLES = {
    "error": ("HATA", "bold red"),
    "warning": ("UYARI", f"bold {MUTED_AMBER}"),
}

HEALTH_STYLES = {
    "ok": ("OK", f"bold {MUTED_GREEN}"),
    "warning": ("UYARI", f"bold {MUTED_AMBER}"),
    "error": ("HATA", "bold red"),
}

ANSWER_MODE_STYLES = {
    "generative": ("Üretken", PRIMARY_BRIGHT),
    "extractive": ("Doğrudan", MUTED_GREEN),
    "fallback_extractive": ("Kaynak metni", MUTED_AMBER),
    "no_evidence": ("Kanıt bulunamadı", "bright_black"),
    "ungrounded": ("Cevap kaynağa dayanmıyor", "bright_black"),
}

CONTENT_PADDING = (0, 1)
PLAIN_INPUT_PROMPT = "> "
READLINE_INPUT_PROMPT = (
    "\001\033[38;2;169;101;107m\002>\001\033[0m\002 "
)


def print_banner(embedding_model, llm_model, project_root=None):
    title = Text("Local RAG Assistant", style=f"bold {PRIMARY_BRIGHT}")
    short_embedding_name = embedding_model.rsplit("/", maxsplit=1)[-1]
    mascot = Text(
        "   ╭─────╮\n"
        "╭──┤ • • ├──╮\n"
        "╰─╮│  ▴  │╭─╯\n"
        "  ╰┴─────┴╯",
        style=PRIMARY_BRIGHT,
    )
    details = Text()
    details.append("Yerel belge yardımcın hazır.\n", style="bold")
    details.append("LLM  ", style="dim")
    details.append(f"{llm_model}\n")
    details.append("Embedding  ", style="dim")
    details.append(short_embedding_name)
    details.append("\nDocs  ", style="dim")
    details.append("docs/")
    details.append("    Database  ", style="dim")
    details.append("data/rag.db")
    if project_root is not None:
        details.append("\nProje  ", style="dim")
        details.append(str(project_root))

    body = Table.grid(expand=True, padding=(0, 2))
    body.add_column(width=16, no_wrap=True)
    body.add_column(ratio=1)
    body.add_row(mascot, details)

    console.print()
    console.print(
        Panel(
            body,
            title=title,
            title_align="left",
            border_style=PRIMARY_DARK,
            padding=(0, 1),
        )
    )
    console.print(
        Text.assemble(
            ("/", f"bold {PRIMARY_BRIGHT}"),
            (" yazınca komutlar açılır; normal soru için doğrudan yaz.", "dim"),
        )
    )


def print_table(title, columns, rows, footer=None):
    table = Table(
        title=title,
        title_style="bold",
        title_justify="left",
        box=box.SIMPLE_HEAVY,
        border_style="bright_black",
        header_style=f"bold {PRIMARY_BRIGHT}",
        show_edge=False,
        pad_edge=False,
        collapse_padding=True,
    )

    for column in columns:
        table.add_column(
            column[0],
            style=column[1] if len(column) > 1 else None,
            justify=column[2] if len(column) > 2 else "left",
            no_wrap=column[3] if len(column) > 3 else False,
            overflow=column[4] if len(column) > 4 else "ellipsis",
        )

    for row in rows:
        table.add_row(*(value if isinstance(value, Text) else str(value) for value in row))

    console.print()
    console.print(Padding(table, CONTENT_PADDING))

    if footer:
        console.print(Padding(Text(footer, style="dim"), CONTENT_PADDING))


def print_health_report(checks):
    rows = []

    for check in checks:
        label, style = HEALTH_STYLES[check.status]
        status = Text(label, style=style)
        detail = Text(check.message)

        if check.solution:
            detail.append(f"\nÇözüm: {check.solution}", style=MUTED_AMBER)

        rows.append((status, check.name, detail))

    print_table(
        "Sistem kontrolü",
        [
            ("Durum", None, "left", True),
            ("Kontrol", "bold", "left", True),
            ("Açıklama",),
        ],
        rows,
    )

    ok_count = sum(check.status == "ok" for check in checks)
    warning_count = sum(check.status == "warning" for check in checks)
    error_count = sum(check.status == "error" for check in checks)
    summary = Text("Sonuç  ", style="bold")
    summary.append(f"{ok_count} başarılı", style=MUTED_GREEN)
    summary.append(f"  {warning_count} uyarı", style=MUTED_AMBER)
    summary.append(f"  {error_count} hata", style="red")
    console.print(summary)


def print_answer(answer, mode, best_score):
    mode_label, mode_style = ANSWER_MODE_STYLES.get(
        mode,
        (mode.replace("_", " ").title(), PRIMARY_BRIGHT),
    )
    title = Text("Cevap", style="bold")
    title.append("  ·  ", style="dim")
    title.append(mode_label, style=f"bold {mode_style}")
    title.append("  ·  Skor ", style="dim")
    title.append(f"{best_score:.4f}", style=mode_style)

    console.print()
    console.print(
        Padding(
            Panel(
                Text(answer.strip()),
                title=title,
                title_align="left",
                border_style=mode_style,
                padding=(1, 1),
            ),
            CONTENT_PADDING,
        )
    )


def print_chunk_detail(chunk):
    metadata = Text()
    metadata.append(f"ID {chunk['id']}", style=f"bold {PRIMARY_BRIGHT}")
    metadata.append("  ·  ", style="dim")
    metadata.append(chunk["source_name"], style="bold")

    if chunk.get("page_number") is not None:
        metadata.append(f"  ·  Sayfa {chunk['page_number']}", style="dim")

    if chunk.get("chunk_index") is not None:
        metadata.append(f"  ·  Parça {chunk['chunk_index']}", style="dim")

    console.print()
    console.print(
        Padding(
            Panel(
                Text(chunk["chunk_text"].strip()),
                title=metadata,
                title_align="left",
                border_style=PRIMARY,
                padding=(1, 1),
            ),
            CONTENT_PADDING,
        )
    )


def print_performance(retrieval_time, generation_time, total_time):
    metrics = Text()
    metrics.append("Arama  ", style="dim")
    metrics.append(f"{retrieval_time:.3f} sn", style=PRIMARY_BRIGHT)
    metrics.append("   ·   Yanıt  ", style="dim")
    metrics.append(f"{generation_time:.3f} sn", style=PRIMARY_BRIGHT)
    metrics.append("   ·   Toplam  ", style="dim")
    metrics.append(f"{total_time:.3f} sn", style="bold")
    console.print(Padding(metrics, CONTENT_PADDING))


def print_success(message):
    console.print(Text.assemble("\n", ("OK", f"bold {MUTED_GREEN}"), "  ", message))


def print_info(message):
    console.print(Text.assemble("\n", ("BİLGİ", f"bold {PRIMARY_BRIGHT}"), "  ", message))


def print_issue(level, message, solution=None, error=None, debug=False):
    if level not in ISSUE_STYLES:
        raise ValueError(f"Bilinmeyen mesaj seviyesi: {level}")

    label, style = ISSUE_STYLES[level]
    console.print(Text.assemble("\n", (label, style), "  ", message))

    if solution:
        console.print(Text.assemble("      ", ("Çözüm  ", "bold"), solution))

    if debug and error is not None:
        console.print(
            Text.assemble(
                "      ",
                ("Teknik detay  ", f"bold {DEBUG_COLOR}"),
                f"{type(error).__name__}: {error}",
            )
        )


def activity(message):
    if not console.is_terminal:
        return nullcontext()

    return console.status(
        Text(message, style=PRIMARY_BRIGHT),
        spinner="dots",
        spinner_style=PRIMARY_BRIGHT,
    )


class RAGProgress:
    STAGES = ("retrieval", "model", "generation")

    def __init__(self, model_name, console_instance=None, live_factory=Live):
        self.model_name = model_name
        self.console = console_instance or console
        self.completed = set()
        self.current_stage = None
        self._status = None
        self._live = None
        self._live_factory = live_factory
        self._streaming = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self._live is not None:
            self._live.stop()
        if self._status is not None:
            self._status.stop()
        return False

    @contextmanager
    def stage(self, stage_name):
        if stage_name not in self.STAGES:
            raise ValueError(f"Bilinmeyen RAG aşaması: {stage_name}")

        self.current_stage = stage_name
        self._refresh()
        try:
            yield
        except Exception:
            raise
        else:
            self.completed.add(stage_name)
            self.current_stage = None
            self._refresh()

    def render(self):
        labels = {
            "retrieval": "Arama",
            "model": self.model_name,
            "generation": "Yanıt",
        }
        active_labels = {
            "retrieval": "Arama yapılıyor",
            "model": f"{self.model_name} hazırlanıyor",
            "generation": "Yanıt üretiliyor",
        }
        content = Text()

        for index, stage_name in enumerate(self.STAGES):
            if index:
                content.append("  ·  ", style="dim")

            if stage_name in self.completed:
                content.append(f"✓ {labels[stage_name]}", style=MUTED_GREEN)
            elif stage_name == self.current_stage:
                content.append(
                    f"● {active_labels[stage_name]}",
                    style=f"bold {PRIMARY_BRIGHT}",
                )
            else:
                content.append(f"○ {labels[stage_name]}", style="dim")

        return content

    def _refresh(self):
        if not self.console.is_terminal or self._streaming:
            return

        if self._status is None:
            self._status = self.console.status(
                self.render(),
                spinner="dots",
                spinner_style=PRIMARY_BRIGHT,
            )
            self._status.start()
        else:
            self._status.update(self.render())

    def update_answer(self, answer):
        if not self.console.is_terminal or not answer:
            return

        if not self._streaming:
            self._streaming = True
            if self._status is not None:
                self._status.stop()
                self._status = None
            self._live = self._live_factory(
                self._stream_panel(answer),
                console=self.console,
                refresh_per_second=12,
                transient=True,
            )
            self._live.start(refresh=True)
            return

        self._live.update(self._stream_panel(answer), refresh=True)

    @staticmethod
    def _stream_panel(answer):
        title = Text("Yanıt üretiliyor", style="bold")
        return Padding(
            Panel(
                Text(answer),
                title=title,
                title_align="left",
                border_style=PRIMARY_BRIGHT,
                padding=(1, 1),
            ),
            CONTENT_PADDING,
        )


def rag_progress(model_name):
    return RAGProgress(model_name)


def read_prompt():
    console.print()
    prompt = (
        READLINE_INPUT_PROMPT
        if console.is_terminal and console.color_system and sys.stdin.isatty()
        else PLAIN_INPUT_PROMPT
    )
    return input(prompt)


def print_submitted_prompt(value):
    console.print()
    prompt = Text("> ", style=f"bold {PRIMARY_BRIGHT}")
    prompt.append(value)
    console.print(prompt)
