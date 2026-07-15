import argparse
import shlex
import time

from app import __version__
from app.cli_output import (
    activity,
    console,
    print_answer,
    print_banner as render_banner,
    print_health_report,
    print_info,
    print_issue,
    print_performance,
    print_success,
    print_table,
    read_prompt,
)
from app.config import (
    CONTEXT_SCORE_THRESHOLD,
    EXTRACTIVE_SCORE_THRESHOLD,
    MAX_EXTRACTIVE_CHARS,
    MIN_GENERATIVE_ANSWER_CHARS,
    SIMILARITY_THRESHOLD,
    TOP_K,
    USE_EXTRACTIVE_FALLBACK,
)
from app.database import DB_PATH, get_chunk_stats, get_indexed_sources
from app.document_manager import (
    DocumentManagementError,
    add_document,
    remove_document,
    resolve_managed_document,
)
from app.embeddings import (
    MODEL_NAME as EMBEDDING_MODEL_NAME,
    get_local_model_path,
    is_embedding_model_loaded,
)
from app.health import check_foundry, run_health_checks
from app.index_state import get_index_freshness
from app.retrieval import get_top_chunks
from app.prompts import build_rag_messages
from app.llm import LocalLLM, MODEL_ALIAS, is_valid_answer
from app.ingest import CHUNK_OVERLAP, CHUNK_SIZE, DOCS_DIR, ingest_documents

DEBUG = False

_llm = None


def get_llm():
    global _llm

    if _llm is None:
        _llm = LocalLLM(show_startup_output=DEBUG)

    return _llm


def print_banner():
    render_banner(EMBEDDING_MODEL_NAME, MODEL_ALIAS)


def print_help():
    print_table(
        "Komutlar",
        [("Komut", "bold cyan", "left", True), ("Açıklama",)],
        [
            ("/help", "Komut listesini gösterir"),
            ("/stats", "İndeks, model ve eşik bilgilerini gösterir"),
            ("/model", "Model, cache ve yüklenme durumunu gösterir"),
            ("/config", "Aktif RAG ayarlarını salt okunur gösterir"),
            ("/sources", "İndeksteki dosya, sayfa ve chunk sayılarını gösterir"),
            ("/doctor", "Sistem bileşenlerinin sağlık durumunu kontrol eder"),
            ("/add <yol>", "TXT veya PDF dosyasını docs/ klasörüne ekler"),
            ("/remove <dosya>", "Dokümanı onay alarak docs/ klasöründen siler"),
            ("/reindex", "docs/ klasörünü yeniden indeksler"),
            ("/debug on", "Teknik debug çıktısını açar"),
            ("/debug off", "Teknik debug çıktısını kapatır"),
            ("/exit", "Uygulamadan çıkar"),
        ],
        footer="Normal soru sormak için doğrudan yazman yeterli.",
    )


def print_model_info():
    checks = check_foundry()
    foundry_check = next(
        (check for check in checks if check.name == "Foundry Local"),
        None,
    )
    model_check = next(
        (check for check in checks if check.name == "LLM modeli"),
        None,
    )
    embedding_cached = get_local_model_path() is not None
    short_embedding_name = EMBEDDING_MODEL_NAME.rsplit("/", maxsplit=1)[-1]

    foundry_status = foundry_check.message if foundry_check else "kontrol edilemedi"
    model_cache_status = model_check.message if model_check else "kontrol edilemedi"
    footer = "Bu komut model yüklemez ve ayar değiştirmez."

    if model_check and model_check.solution:
        footer += f" Çözüm: {model_check.solution}"

    print_table(
        "Model durumu",
        [
            ("Bileşen", "bold", "left", True),
            ("Değer", "cyan", "left", False, "fold"),
            ("Durum",),
        ],
        [
            ("Chat modeli", MODEL_ALIAS, "aktif alias"),
            ("Çalışma zamanı", "Microsoft Foundry Local", foundry_status),
            ("Chat model cache", MODEL_ALIAS, model_cache_status),
            (
                "Chat model oturumu",
                "yüklü" if _llm is not None else "henüz yüklenmedi",
                "lazy-load: ilk üretken cevapta yüklenir",
            ),
            ("Embedding modeli", short_embedding_name, "384 boyut"),
            (
                "Embedding cache",
                "hazır" if embedding_cached else "bulunamadı",
                "yerel Hugging Face snapshot",
            ),
            (
                "Embedding oturumu",
                "yüklü" if is_embedding_model_loaded() else "henüz yüklenmedi",
                "lazy-load: ilk aramada yüklenir",
            ),
        ],
        footer=footer,
    )


def print_config_info():
    print_table(
        "RAG yapılandırması",
        [
            ("Ayar", "bold", "left", True),
            ("Değer", "cyan", "right", True),
            ("Açıklama",),
        ],
        [
            ("TOP_K", TOP_K, "Retrieval sonucunda tutulacak en iyi chunk sayısı"),
            (
                "SIMILARITY_THRESHOLD",
                SIMILARITY_THRESHOLD,
                "Altında kalan sorular kapsam dışı kabul edilir",
            ),
            (
                "CONTEXT_SCORE_THRESHOLD",
                CONTEXT_SCORE_THRESHOLD,
                "LLM context'ine girecek minimum chunk skoru",
            ),
            (
                "EXTRACTIVE_SCORE_THRESHOLD",
                EXTRACTIVE_SCORE_THRESHOLD,
                "Doğrudan kaynak cevabı için minimum skor",
            ),
            (
                "USE_EXTRACTIVE_FALLBACK",
                "açık" if USE_EXTRACTIVE_FALLBACK else "kapalı",
                "Güvenli doğrudan/fallback cevabını etkinleştirir",
            ),
            (
                "MAX_EXTRACTIVE_CHARS",
                MAX_EXTRACTIVE_CHARS,
                "Doğrudan gösterilebilecek en uzun kaynak metni",
            ),
            (
                "MIN_GENERATIVE_ANSWER_CHARS",
                MIN_GENERATIVE_ANSWER_CHARS,
                "Daha kısa LLM cevapları geçersiz sayılır",
            ),
            ("CHUNK_SIZE", CHUNK_SIZE, "Bir chunk'ın hedef maksimum karakteri"),
            ("CHUNK_OVERLAP", CHUNK_OVERLAP, "Ardışık chunklar arasındaki tekrar"),
            ("DOCS_DIR", DOCS_DIR, "İndekslenecek doküman klasörü"),
            ("DB_PATH", DB_PATH, "Üretilen SQLite indeks yolu"),
        ],
        footer="Salt okunur görünüm; bu komut ayarları değiştirmez.",
    )


def print_stats():
    stats = get_chunk_stats()
    freshness = get_index_freshness(DOCS_DIR, DB_PATH)
    short_embedding_name = EMBEDDING_MODEL_NAME.rsplit("/", maxsplit=1)[-1]

    print_table(
        "Sistem durumu",
        [("Ayar", "bold"), ("Değer", "cyan")],
        [
            ("Chunk sayısı", stats["total_chunks"]),
            ("Kaynak dosya", stats["source_count"]),
            ("İndeks durumu", freshness.display_status()),
            ("Veritabanı", stats["db_path"]),
            ("Embedding", short_embedding_name),
            ("LLM", MODEL_ALIAS),
            ("Debug", "açık" if DEBUG else "kapalı"),
            ("Top K", TOP_K),
            ("Chunk size / overlap", f"{CHUNK_SIZE} / {CHUNK_OVERLAP}"),
            ("Similarity threshold", SIMILARITY_THRESHOLD),
            ("Context threshold", CONTEXT_SCORE_THRESHOLD),
        ],
    )


def warn_if_index_is_stale():
    freshness = get_index_freshness(DOCS_DIR, DB_PATH)

    if freshness.status == "stale":
        print_issue(
            "warning",
            f"İndeks güncel değil. {freshness.change_summary()}",
            solution="/reindex veya local-rag reindex çalıştır.",
        )
    elif freshness.status == "untracked":
        print_issue(
            "warning",
            "İndeksin hangi dokümanlardan üretildiği bilinmiyor.",
            solution="/reindex veya local-rag reindex çalıştır.",
        )
    elif freshness.status == "error":
        print_issue(
            "warning",
            "Doküman değişiklikleri kontrol edilemedi.",
            solution="/doctor çalıştır.",
            error=RuntimeError(freshness.error),
            debug=DEBUG,
        )

    return freshness


def print_indexed_sources():
    sources = get_indexed_sources()

    if not sources:
        print_issue(
            "warning",
            "İndekste kaynak dosya yok.",
            solution="/reindex çalıştır.",
        )
        return

    total_chunks = sum(source["chunk_count"] for source in sources)

    rows = []
    for source in sources:
        source_type = source["source_type"] or "bilinmiyor"
        page_count = source["page_count"]
        rows.append(
            (
                source["source_name"],
                source_type,
                page_count if page_count > 0 else "-",
                source["chunk_count"],
            )
        )

    print_table(
        "İndeksteki kaynaklar",
        [
            ("Dosya", "bold"),
            ("Tür", "cyan", "left", True),
            ("Sayfa", None, "right", True),
            ("Chunk", None, "right", True),
        ],
        rows,
        footer=f"Toplam: {len(sources)} dosya, {total_chunks} chunk",
    )


def print_doctor_report():
    checks = run_health_checks()
    print_health_report(checks)


def print_sources(chunks):
    rows = []
    for chunk in chunks:
        rows.append(
            (
                chunk["source_name"],
                chunk.get("page_number") if chunk.get("page_number") is not None else "-",
                chunk.get("chunk_index") if chunk.get("chunk_index") is not None else "-",
                chunk["id"],
                f"{chunk['score']:.4f}",
            )
        )

    print_table(
        "Kaynaklar",
        [
            ("Dosya", "bold"),
            ("Sayfa", None, "right", True),
            ("Parça", None, "right", True),
            ("ID", "dim", "right", True),
            ("Skor", "cyan", "right", True),
        ],
        rows,
    )


def print_debug_info(question, chunks, messages):
    console.rule("[bold magenta]DEBUG[/bold magenta]", style="magenta")
    console.print("[bold]Kullanıcı sorusu[/bold]")
    console.print(question)
    console.print("\n[bold]Retrieved chunks[/bold]")
    for chunk in chunks:
        console.print(
            f"[dim]ID {chunk['id']} · {chunk['source_name']} · "
            f"skor {chunk['score']:.4f}[/dim]"
        )
        console.print(chunk["chunk_text"])

    console.print("\n[bold]Modele gönderilen mesajlar[/bold]")
    for message in messages:
        console.print(f"[bold magenta]{message['role']}[/bold magenta]")
        console.print(message["content"])

    console.rule(style="magenta")


def should_use_extractive_answer(context_chunks):
    if not USE_EXTRACTIVE_FALLBACK:
        return False

    if len(context_chunks) != 1:
        return False

    best_chunk = context_chunks[0]

    if best_chunk["score"] < EXTRACTIVE_SCORE_THRESHOLD:
        return False

    if len(best_chunk["chunk_text"]) > MAX_EXTRACTIVE_CHARS:
        return False

    return True


def get_fallback_answer(context_chunks):
    return context_chunks[0]["chunk_text"].strip()


def generate_with_fallback(messages, context_chunks, llm=None):
    fallback_answer = get_fallback_answer(context_chunks)

    try:
        llm_client = llm or get_llm()
        generated_answer = llm_client.generate_answer(messages)
    except Exception as error:
        return (
            fallback_answer,
            "fallback_extractive",
            "LLM yanıtı alınamadı; kaynak metin kullanıldı.",
            error,
        )

    if not is_valid_answer(generated_answer):
        return (
            fallback_answer,
            "fallback_extractive",
            "LLM cevabı yeterli bulunmadı; kaynak metin kullanıldı.",
            None,
        )

    return generated_answer, "generative", None, None


def run_command_safely(action, error_message, solution):
    try:
        action()
    except Exception as error:
        print_issue(
            "error",
            error_message,
            solution=solution,
            error=error,
            debug=DEBUG,
        )
        return False

    return True


def reindex_documents():
    try:
        with activity("Dokümanlar yeniden indeksleniyor..."):
            total_chunks = ingest_documents()
    except Exception as error:
        print_issue(
            "error",
            "Re-index tamamlanamadı; mevcut indeks korundu.",
            solution="/doctor çalıştır ve docs/ klasöründeki dosyaları kontrol et.",
            error=error,
            debug=DEBUG,
        )
        return False

    print_success(f"Re-index tamamlandı · {total_chunks} chunk")
    return True


def add_document_command(source_path):
    try:
        with activity("Doküman doğrulanıyor ve ekleniyor..."):
            destination = add_document(source_path, DOCS_DIR)
    except DocumentManagementError as error:
        print_issue(
            "error",
            str(error),
            solution=(
                "Dosya yolunu, türünü, içeriğini ve docs/ içindeki mevcut adları kontrol et."
            ),
        )
        return False
    except Exception as error:
        print_issue(
            "error",
            "Doküman eklenemedi.",
            solution="Dosya izinlerini ve docs/ klasörünü kontrol et.",
            error=error,
            debug=DEBUG,
        )
        return False

    print_success(f"Doküman eklendi · {destination.name}")
    print_info("İndeksi güncellemek için /reindex veya local-rag reindex çalıştır.")
    return True


def confirm_document_removal(source_name):
    try:
        answer = console.input(
            f"\n[bold yellow]{source_name} silinsin mi?[/bold yellow] "
            "[dim](e/H)[/dim] "
        )
    except (EOFError, KeyboardInterrupt):
        return False

    return answer.strip().lower() in {"e", "evet"}


def remove_document_command(source_name, assume_yes=False):
    try:
        destination = resolve_managed_document(source_name, DOCS_DIR)
    except DocumentManagementError as error:
        print_issue(
            "error",
            str(error),
            solution="Dosya adlarını görmek için /sources veya docs/ klasörünü kontrol et.",
        )
        return False

    if not assume_yes and not confirm_document_removal(destination.name):
        print_info("Silme işlemi iptal edildi.")
        return True

    try:
        removed_path = remove_document(destination.name, DOCS_DIR)
    except DocumentManagementError as error:
        print_issue("error", str(error), solution="docs/ klasörünü kontrol et.")
        return False
    except Exception as error:
        print_issue(
            "error",
            "Doküman silinemedi.",
            solution="Dosya izinlerini ve docs/ klasörünü kontrol et.",
            error=error,
            debug=DEBUG,
        )
        return False

    print_success(f"Doküman silindi · {removed_path.name}")
    print_info("İndeksi güncellemek için /reindex veya local-rag reindex çalıştır.")
    return True


INFO_COMMAND_MESSAGES = {
    "/stats": (
        "Sistem bilgileri okunamadı.",
        "/doctor çalıştır.",
    ),
    "/model": (
        "Model bilgileri okunamadı.",
        "/doctor çalıştır.",
    ),
    "/config": (
        "RAG yapılandırması gösterilemedi.",
        "/debug on ile teknik ayrıntıları açıp yeniden dene.",
    ),
    "/sources": (
        "İndekslenen kaynaklar okunamadı.",
        "/doctor çalıştır; gerekirse /reindex ile indeksi yenile.",
    ),
    "/doctor": (
        "Sistem kontrolü tamamlanamadı.",
        "/debug on ile teknik ayrıntıları açıp yeniden dene.",
    ),
}


def execute_command(command):
    if command == "/reindex":
        return reindex_documents()

    actions = {
        "/stats": print_stats,
        "/model": print_model_info,
        "/config": print_config_info,
        "/sources": print_indexed_sources,
        "/doctor": print_doctor_report,
    }
    error_message, solution = INFO_COMMAND_MESSAGES[command]
    action = actions[command]
    return run_command_safely(action, error_message, solution)


def handle_command(command_line):
    global DEBUG

    stripped_command = command_line.strip()
    normalized_command = stripped_command.lower()
    leading_command = (
        normalized_command.split(maxsplit=1)[0]
        if normalized_command
        else ""
    )

    if normalized_command in ["/exit", "/quit", "q", "quit", "exit"]:
        return "exit"

    if normalized_command == "/help":
        print_help()
        return "handled"

    if normalized_command in INFO_COMMAND_MESSAGES:
        execute_command(normalized_command)
        return "handled"

    if normalized_command == "/reindex":
        execute_command(normalized_command)
        return "handled"

    if normalized_command == "/debug on":
        DEBUG = True
        print_info("Debug modu açıldı.")
        return "handled"

    if normalized_command == "/debug off":
        DEBUG = False
        print_info("Debug modu kapatıldı.")
        return "handled"

    if leading_command in {"/add", "/remove"}:
        try:
            arguments = shlex.split(stripped_command)
        except ValueError as error:
            print_issue(
                "error",
                "Komut argümanları okunamadı.",
                solution="Boşluk içeren yolları çift tırnak içine al.",
                error=error,
                debug=DEBUG,
            )
            return "handled"

        command_name = arguments[0].lower()

        if command_name not in {"/add", "/remove"} or len(arguments) != 2:
            usage = (
                "/add <dosya-yolu>"
                if command_name == "/add"
                else "/remove <dosya-adı>"
            )
            print_issue(
                "warning",
                "Komut eksik veya hatalı.",
                solution=f"Kullanım: {usage}",
            )
            return "handled"

        if command_name == "/add":
            add_document_command(arguments[1])
        else:
            remove_document_command(arguments[1])
        return "handled"

    if normalized_command.startswith("/"):
        print_issue(
            "warning",
            "Bilinmeyen komut.",
            solution="Komutları görmek için /help yaz.",
        )
        return "handled"

    return None


def answer_question(question):
    question = question.strip()

    if not question:
        print_issue("warning", "Soru boş olamaz.")
        return False

    warn_if_index_is_stale()

    total_start_time = time.perf_counter()
    retrieval_start_time = time.perf_counter()

    try:
        with activity("İlgili kaynaklar aranıyor..."):
            chunks = get_top_chunks(question, top_k=TOP_K)
    except Exception as error:
        print_issue(
            "error",
            "Dokümanlarda arama yapılamadı.",
            solution="/doctor çalıştır; indeks sorunu varsa /reindex ile yenile.",
            error=error,
            debug=DEBUG,
        )
        return False

    retrieval_end_time = time.perf_counter()
    retrieval_time = retrieval_end_time - retrieval_start_time

    if not chunks:
        print_issue(
            "warning",
            "Aranabilecek bir indeks bulunamadı.",
            solution="/reindex çalıştır.",
        )
        return False

    best_score = chunks[0]["score"]

    if best_score < SIMILARITY_THRESHOLD:
        total_time = time.perf_counter() - total_start_time
        print_answer(
            "Bu bilgi verilen dokümanlarda yok.",
            "no_evidence",
            best_score,
        )
        print_performance(retrieval_time, 0.0, total_time)
        return True

    context_chunks = [
        chunk
        for chunk in chunks
        if chunk["score"] >= CONTEXT_SCORE_THRESHOLD
    ]

    if not context_chunks:
        context_chunks = [chunks[0]]

    messages = build_rag_messages(question, context_chunks)

    if DEBUG:
        print_debug_info(question, context_chunks, messages)

    generation_start_time = time.perf_counter()

    if should_use_extractive_answer(context_chunks):
        answer = context_chunks[0]["chunk_text"]
        answer_mode = "extractive"
    else:
        activity_message = (
            f"{MODEL_ALIAS} yükleniyor ve cevap hazırlanıyor..."
            if _llm is None
            else "Cevap hazırlanıyor..."
        )

        with activity(activity_message):
            answer, answer_mode, fallback_notice, llm_error = generate_with_fallback(
                messages,
                context_chunks,
            )

        if fallback_notice:
            print_issue(
                "warning",
                fallback_notice,
                solution="/doctor ile LLM durumunu kontrol et." if llm_error else None,
                error=llm_error,
                debug=DEBUG,
            )

    generation_time = time.perf_counter() - generation_start_time
    total_time = time.perf_counter() - total_start_time

    print_answer(answer, answer_mode, best_score)
    print_sources(context_chunks)
    print_performance(retrieval_time, generation_time, total_time)
    return True


def main():
    print_banner()

    while True:
        try:
            question = read_prompt().strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Oturum kapatıldı.[/dim]")
            break

        if not question:
            continue

        command_result = handle_command(question)

        if command_result == "exit":
            break

        if command_result == "handled":
            continue

        answer_question(question)


class TurkishArgumentParser(argparse.ArgumentParser):
    def __init__(self, *args, **kwargs):
        kwargs["add_help"] = False
        super().__init__(*args, **kwargs)
        self._positionals.title = "argümanlar"
        self._optionals.title = "seçenekler"
        self.add_argument(
            "-h",
            "--help",
            action="help",
            help="Bu yardım metnini gösterir ve çıkar.",
        )

    def format_usage(self):
        return super().format_usage().replace("usage:", "kullanım:", 1)

    def format_help(self):
        return super().format_help().replace("usage:", "kullanım:", 1)


def build_cli_parser():
    parser = TurkishArgumentParser(
        prog="local-rag",
        description="Yerel dokümanlarından Türkçe cevap üreten RAG asistanı.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Teknik retrieval ve hata ayrıntılarını gösterir.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
        help="Sürüm bilgisini gösterir ve çıkar.",
    )

    subparsers = parser.add_subparsers(dest="command", title="komutlar")
    ask_parser = subparsers.add_parser(
        "ask",
        help="Tek bir soru sorar ve işlem tamamlanınca çıkar.",
    )
    ask_parser.add_argument("question", nargs="+", help="Sorulacak metin")

    add_parser = subparsers.add_parser(
        "add",
        help="TXT veya PDF dosyasını docs/ klasörüne ekler.",
    )
    add_parser.add_argument("path", help="Eklenecek dosyanın yolu")

    remove_parser = subparsers.add_parser(
        "remove",
        help="Dokümanı onay alarak docs/ klasöründen siler.",
    )
    remove_parser.add_argument("source_name", help="docs/ içindeki dosya adı")
    remove_parser.add_argument(
        "--yes",
        action="store_true",
        help="Onay sorusunu atlayarak siler.",
    )

    command_help = {
        "reindex": "docs/ klasörünü yeniden indeksler.",
        "stats": "İndeks ve sistem durumunu gösterir.",
        "sources": "İndeksteki kaynakları gösterir.",
        "doctor": "Sistem bileşenlerini kontrol eder.",
        "model": "Model, cache ve lazy-load durumunu gösterir.",
        "config": "Aktif RAG ayarlarını gösterir.",
    }

    for command, help_text in command_help.items():
        subparsers.add_parser(command, help=help_text)

    return parser


def cli(argv=None):
    global DEBUG

    args = build_cli_parser().parse_args(argv)
    DEBUG = args.debug

    if args.command is None:
        main()
        return 0

    if args.command == "ask":
        question = " ".join(args.question)
        return 0 if answer_question(question) else 1

    if args.command == "add":
        return 0 if add_document_command(args.path) else 1

    if args.command == "remove":
        return 0 if remove_document_command(args.source_name, args.yes) else 1

    return 0 if execute_command(f"/{args.command}") else 1


if __name__ == "__main__":
    raise SystemExit(cli())
