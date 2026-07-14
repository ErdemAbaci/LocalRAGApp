import time

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
from app.embeddings import (
    MODEL_NAME as EMBEDDING_MODEL_NAME,
    get_local_model_path,
    is_embedding_model_loaded,
)
from app.health import check_foundry, run_health_checks
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
    short_embedding_name = EMBEDDING_MODEL_NAME.rsplit("/", maxsplit=1)[-1]

    print_table(
        "Sistem durumu",
        [("Ayar", "bold"), ("Değer", "cyan")],
        [
            ("Chunk sayısı", stats["total_chunks"]),
            ("Kaynak dosya", stats["source_count"]),
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


def handle_command(command):
    global DEBUG

    if command in ["/exit", "/quit", "q", "quit", "exit"]:
        return "exit"

    if command == "/help":
        print_help()
        return "handled"

    if command == "/stats":
        run_command_safely(
            print_stats,
            "Sistem bilgileri okunamadı.",
            "/doctor çalıştır.",
        )
        return "handled"

    if command == "/model":
        run_command_safely(
            print_model_info,
            "Model bilgileri okunamadı.",
            "/doctor çalıştır.",
        )
        return "handled"

    if command == "/config":
        run_command_safely(
            print_config_info,
            "RAG yapılandırması gösterilemedi.",
            "/debug on ile teknik ayrıntıları açıp yeniden dene.",
        )
        return "handled"

    if command == "/sources":
        run_command_safely(
            print_indexed_sources,
            "İndekslenen kaynaklar okunamadı.",
            "/doctor çalıştır; gerekirse /reindex ile indeksi yenile.",
        )
        return "handled"

    if command == "/doctor":
        run_command_safely(
            print_doctor_report,
            "Sistem kontrolü tamamlanamadı.",
            "/debug on ile teknik ayrıntıları açıp yeniden dene.",
        )
        return "handled"

    if command == "/reindex":
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
        else:
            print_success(f"Re-index tamamlandı · {total_chunks} chunk")

        return "handled"

    if command == "/debug on":
        DEBUG = True
        print_info("Debug modu açıldı.")
        return "handled"

    if command == "/debug off":
        DEBUG = False
        print_info("Debug modu kapatıldı.")
        return "handled"

    if command.startswith("/"):
        print_issue(
            "warning",
            "Bilinmeyen komut.",
            solution="Komutları görmek için /help yaz.",
        )
        return "handled"

    return None


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

        command_result = handle_command(question.lower())

        if command_result == "exit":
            break

        if command_result == "handled":
            continue
        
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
            continue

        retrieval_end_time = time.perf_counter()

        retrieval_time = retrieval_end_time - retrieval_start_time

        if not chunks:
            print_issue(
                "warning",
                "Aranabilecek bir indeks bulunamadı.",
                solution="/reindex çalıştır.",
            )
            continue

        best_score = chunks[0]["score"]

        if best_score < SIMILARITY_THRESHOLD:
            total_end_time = time.perf_counter()
            total_time = total_end_time - total_start_time

            print_answer(
                "Bu bilgi verilen dokümanlarda yok.",
                "no_evidence",
                best_score,
            )
            print_performance(retrieval_time, 0.0, total_time)
            continue
        
        context_chunks = [chunk for chunk in chunks if chunk["score"] >= CONTEXT_SCORE_THRESHOLD]
        if not context_chunks:
            context_chunks = [chunks[0]]
        messages = build_rag_messages(question, context_chunks)

        if DEBUG:
            print_debug_info(question, context_chunks, messages)

        if should_use_extractive_answer(context_chunks):
            generation_start_time = time.perf_counter()
            answer = context_chunks[0]["chunk_text"]
            answer_mode = "extractive"
            generation_end_time = time.perf_counter()
        else:
            generation_start_time = time.perf_counter()
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

            generation_end_time = time.perf_counter()

        generation_time = generation_end_time - generation_start_time

        total_end_time = time.perf_counter()
        total_time = total_end_time - total_start_time

        print_answer(answer, answer_mode, best_score)
        print_sources(context_chunks)
        print_performance(retrieval_time, generation_time, total_time)


if __name__ == "__main__":
    main()
