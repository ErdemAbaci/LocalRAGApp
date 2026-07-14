from contextlib import nullcontext

from rich import box
from rich.console import Console
from rich.padding import Padding
from rich.panel import Panel
from rich.table import Table
from rich.text import Text


console = Console(highlight=False)

ISSUE_STYLES = {
    "error": ("HATA", "bold red"),
    "warning": ("UYARI", "bold yellow"),
}

HEALTH_STYLES = {
    "ok": ("OK", "bold green"),
    "warning": ("UYARI", "bold yellow"),
    "error": ("HATA", "bold red"),
}

ANSWER_MODE_STYLES = {
    "generative": ("Üretken", "cyan"),
    "extractive": ("Doğrudan", "green"),
    "fallback_extractive": ("Kaynak metni", "yellow"),
    "no_evidence": ("Kanıt bulunamadı", "bright_black"),
}

CONTENT_PADDING = (0, 1)


def print_banner(embedding_model, llm_model):
    title = Text("Local RAG Assistant", style="bold cyan")
    short_embedding_name = embedding_model.rsplit("/", maxsplit=1)[-1]
    body = Text()
    body.append("Yerel doküman soru-cevap asistanı\n", style="bold")
    body.append("LLM  ", style="dim")
    body.append(f"{llm_model}\n")
    body.append("Embedding  ", style="dim")
    body.append(short_embedding_name)
    body.append("\nDocs  ", style="dim")
    body.append("docs/")
    body.append("    Database  ", style="dim")
    body.append("data/rag.db")

    console.print()
    console.print(
        Panel(
            body,
            title=title,
            title_align="left",
            border_style="cyan",
            padding=(0, 1),
        )
    )
    console.print("[dim]Komutlar için [bold]/help[/bold] yaz.[/dim]")


def print_table(title, columns, rows, footer=None):
    table = Table(
        title=title,
        title_style="bold",
        title_justify="left",
        box=box.SIMPLE_HEAVY,
        border_style="bright_black",
        header_style="bold cyan",
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
            detail.append(f"\nÇözüm: {check.solution}", style="yellow")

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
    summary.append(f"{ok_count} başarılı", style="green")
    summary.append(f"  {warning_count} uyarı", style="yellow")
    summary.append(f"  {error_count} hata", style="red")
    console.print(summary)


def print_answer(answer, mode, best_score):
    mode_label, mode_style = ANSWER_MODE_STYLES.get(
        mode,
        (mode.replace("_", " ").title(), "cyan"),
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


def print_performance(retrieval_time, generation_time, total_time):
    metrics = Text()
    metrics.append("Arama  ", style="dim")
    metrics.append(f"{retrieval_time:.3f} sn", style="cyan")
    metrics.append("   ·   Yanıt  ", style="dim")
    metrics.append(f"{generation_time:.3f} sn", style="cyan")
    metrics.append("   ·   Toplam  ", style="dim")
    metrics.append(f"{total_time:.3f} sn", style="bold")
    console.print(Padding(metrics, CONTENT_PADDING))


def print_success(message):
    console.print(Text.assemble("\n", ("OK", "bold green"), "  ", message))


def print_info(message):
    console.print(Text.assemble("\n", ("BİLGİ", "bold cyan"), "  ", message))


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
                ("Teknik detay  ", "bold magenta"),
                f"{type(error).__name__}: {error}",
            )
        )


def activity(message):
    if not console.is_terminal:
        return nullcontext()

    return console.status(
        Text(message, style="cyan"),
        spinner="dots",
        spinner_style="cyan",
    )


def read_prompt():
    return console.input("\n[bold cyan]rag>[/bold cyan] ")
