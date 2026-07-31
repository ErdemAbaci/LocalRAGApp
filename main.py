import argparse
import shlex

import app.database as database_module
import app.health as health_module
import app.ingest as ingest_module
from app import __version__
from app.cli_output import (
    DEBUG_COLOR,
    MUTED_AMBER,
    PRIMARY_BRIGHT,
    rag_progress,
    activity,
    console,
    print_answer,
    print_banner as render_banner,
    print_chunk_detail,
    print_health_report,
    print_info,
    print_issue,
    print_performance,
    print_submitted_prompt,
    print_success,
    print_table,
    read_prompt,
)
from app.cli_input import CLIInputManager, CLIStatus
from app.config import (
    CONTEXT_RELATIVE_SCORE_MARGIN,
    CONTEXT_SCORE_THRESHOLD,
    CONTEXT_TERM_EVIDENCE_MIN,
    EXTRACTIVE_SCORE_THRESHOLD,
    MAX_CONTEXT_CHUNKS,
    MAX_EXTRACTIVE_CHARS,
    MIN_GENERATIVE_ANSWER_CHARS,
    NEIGHBOR_CHUNK_RADIUS,
    RRF_K,
    SIMILARITY_THRESHOLD,
    TERM_EVIDENCE_MIN_PREFIX,
    TERM_EVIDENCE_MIN_SHORT_ROOT,
    TERM_EVIDENCE_THRESHOLD,
    TOP_K,
    USE_EXTRACTIVE_FALLBACK,
    USE_HYBRID_SEARCH,
)
from app.database import (
    DB_PATH,
    get_chunk_by_id,
    get_chunk_stats,
    get_indexed_sources,
)
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
from app.llm import (
    DEFAULT_MODEL_ALIAS,
    LocalLLM,
    MODEL_ALIAS,
    get_model_alias_source,
)
from app.ingest import CHUNK_OVERLAP, CHUNK_SIZE, DOCS_DIR, ingest_documents
from app.project import (
    PROJECT_ENV_VAR,
    ProjectConfigurationError,
    get_project_paths,
)
from app.rag_service import EmptyIndexError, EmptyQuestionError, RAGService
from app.session import SessionExportError, SessionHistory

DEBUG = False

_llm = None
_source_filter = None
_session_history = SessionHistory()
PROJECT_PATHS = get_project_paths()


def configure_project(project_path=None, environ=None):
    global DB_PATH, DOCS_DIR, PROJECT_PATHS, _source_filter

    paths = get_project_paths(project_path, environ=environ)
    PROJECT_PATHS = paths
    DB_PATH = paths.db_path
    DOCS_DIR = paths.docs_dir
    database_module.DB_PATH = paths.db_path
    ingest_module.DOCS_DIR = paths.docs_dir
    health_module.DOCS_DIR = paths.docs_dir
    _source_filter = None
    return paths


def get_llm():
    global _llm

    if _llm is None:
        _llm = LocalLLM(show_startup_output=DEBUG)

    return _llm


def print_banner():
    render_banner(
        EMBEDDING_MODEL_NAME,
        MODEL_ALIAS,
        project_root=PROJECT_PATHS.root,
    )


def get_cli_status():
    try:
        source_count = get_chunk_stats()["source_count"]
    except Exception:
        source_count = None

    try:
        freshness = get_index_freshness(DOCS_DIR, DB_PATH)
        index_labels = {
            "current": "indeks güncel",
            "stale": "indeks güncel değil",
            "untracked": "indeks izlenmiyor",
            "missing": "indeks yok",
            "error": "indeks kontrol edilemedi",
        }
        index_status = freshness.status
        index_label = index_labels.get(index_status, "indeks bilinmiyor")
    except Exception:
        index_status = "error"
        index_label = "indeks kontrol edilemedi"

    return CLIStatus(
        model_name=MODEL_ALIAS,
        source_count=source_count,
        index_label=index_label,
        index_status=index_status,
        source_filter=_source_filter,
    )


def print_help():
    print_table(
        "Komutlar",
        [("Komut", f"bold {PRIMARY_BRIGHT}", "left", True), ("Açıklama",)],
        [
            ("/help", "Komut listesini gösterir"),
            ("/stats", "İndeks, model ve eşik bilgilerini gösterir"),
            ("/model", "Model, cache ve yüklenme durumunu gösterir"),
            ("/config", "Aktif RAG ayarlarını salt okunur gösterir"),
            ("/sources", "İndeksteki dosya, sayfa ve chunk sayılarını gösterir"),
            ("/show <chunk-id>", "İndeksteki chunk metnini gösterir"),
            ("/filter <dosya|off>", "Oturumdaki aramayı bir kaynakla sınırlar"),
            ("/ask [--source dosya] <soru>", "İsteğe bağlı kaynak filtresiyle sorar"),
            ("/history", "Bu oturumdaki soru ve cevapları listeler"),
            ("/repeat [id]", "Son veya seçilen soruyu yeniden çalıştırır"),
            ("/export <markdown|json> [yol]", "Oturumu dosyaya aktarır"),
            ("/doctor", "Sistem bileşenlerinin sağlık durumunu kontrol eder"),
            ("/add <yol>", "TXT veya PDF dosyasını docs/ klasörüne ekler"),
            ("/remove <dosya>", "Dokümanı onay alarak docs/ klasöründen siler"),
            ("/benchmark [model]", "Modellerin süre ve kalitesini karşılaştırır"),
            ("/reindex", "docs/ klasörünü yeniden indeksler"),
            ("/debug on", "Teknik debug çıktısını açar"),
            ("/debug off", "Teknik debug çıktısını kapatır"),
            ("/exit", "Uygulamadan çıkar"),
        ],
        footer="Normal soru sormak için doğrudan yazman yeterli.",
    )


def get_session_ids():
    return _session_history.ids()


def print_session_history():
    if not _session_history.entries:
        print_info("Bu oturumda henüz cevaplanmış soru yok.")
        return True

    mode_labels = {
        "generative": "Üretken",
        "extractive": "Doğrudan",
        "fallback_extractive": "Kaynak metni",
        "no_evidence": "Kanıt yok",
    }
    rows = []
    for entry in _session_history.entries:
        rows.append((
            entry.id,
            entry.question,
            mode_labels.get(entry.mode, entry.mode),
            entry.source_filter or "-",
            f"{entry.timings['total_seconds']:.3f} sn",
        ))

    print_table(
        "Oturum geçmişi",
        [
            ("ID", "dim", "right", True),
            ("Soru", "bold", "left", False, "fold"),
            ("Mod", PRIMARY_BRIGHT, "left", True),
            ("Filtre", None, "left", True),
            ("Toplam", None, "right", True),
        ],
        rows,
        footer="/repeat [id] ile bir soruyu yeniden çalıştırabilirsin.",
    )
    return True


def repeat_session_entry(entry_id=None):
    entry = _session_history.get(entry_id)
    if entry is None:
        message = (
            "Bu oturumda tekrarlanabilecek soru yok."
            if entry_id is None
            else f"Oturum geçmişinde {entry_id} numaralı kayıt yok."
        )
        print_issue("warning", message, solution="/history ile kayıtları kontrol et.")
        return False

    print_info(f"{entry.id}. soru yeniden çalıştırılıyor: {entry.question}")
    return answer_question(
        entry.question,
        source_name=entry.source_filter,
        use_active_filter=False,
    )


def export_session(export_format, output_path=None):
    try:
        destination = _session_history.export(
            export_format,
            PROJECT_PATHS.session_export_dir,
            output_path=output_path,
        )
    except SessionExportError as error:
        print_issue(
            "warning",
            str(error),
            solution="Kullanım: /export <markdown|json> [dosya-yolu]",
        )
        return False

    print_success(f"Oturum dışa aktarıldı · {destination}")
    return True


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
            ("Değer", PRIMARY_BRIGHT, "left", False, "fold"),
            ("Durum",),
        ],
        [
            ("Chat modeli", MODEL_ALIAS, "aktif alias"),
            (
                "Model seçimi",
                get_model_alias_source(),
                f"varsayılan: {DEFAULT_MODEL_ALIAS}",
            ),
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
            ("Değer", PRIMARY_BRIGHT, "right", True),
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
                "CONTEXT_RELATIVE_SCORE_MARGIN",
                CONTEXT_RELATIVE_SCORE_MARGIN,
                "En iyi sonuca göre izin verilen maksimum skor farkı",
            ),
            (
                "NEIGHBOR_CHUNK_RADIUS",
                NEIGHBOR_CHUNK_RADIUS,
                "Üretken cevapta eşleşme çevresinden alınacak parça yarıçapı",
            ),
            (
                "MAX_CONTEXT_CHUNKS",
                MAX_CONTEXT_CHUNKS,
                "Modele gönderilecek eşleşme ve komşuların toplam üst sınırı",
            ),
            (
                "CONTEXT_TERM_EVIDENCE_MIN",
                CONTEXT_TERM_EVIDENCE_MIN,
                "İkinci ve sonraki sıraların context'e girmesi için gereken kelime kanıtı",
            ),
            (
                "USE_HYBRID_SEARCH",
                "açık" if USE_HYBRID_SEARCH else "kapalı",
                "Anlam benzerliğine kelime örtüşmesi sıralamasını ekler",
            ),
            (
                "RRF_K",
                RRF_K,
                "İki sıralamayı birleştirirken sıra farklarının etkisini ayarlar",
            ),
            (
                "TERM_EVIDENCE_THRESHOLD",
                TERM_EVIDENCE_THRESHOLD,
                "Soru kelimelerinin context'te bulunması gereken en düşük ağırlıklı oran",
            ),
            (
                "TERM_EVIDENCE_MIN_PREFIX",
                TERM_EVIDENCE_MIN_PREFIX,
                "Türkçe ek eşleştirmesi için gereken en kısa ortak kök",
            ),
            (
                "TERM_EVIDENCE_MIN_SHORT_ROOT",
                TERM_EVIDENCE_MIN_SHORT_ROOT,
                "Kısa köklerde tam kapsanma kuralının geçerli olduğu en kısa uzunluk",
            ),
            (
                "CONTEXT_TERM_EVIDENCE_MIN",
                CONTEXT_TERM_EVIDENCE_MIN,
                "İkinci ve sonraki sıraların context'e girmesi için gereken kelime kanıtı",
            ),
            (
                "USE_HYBRID_SEARCH",
                "açık" if USE_HYBRID_SEARCH else "kapalı",
                "Anlam araması yanında kelime aramasını (BM25) da çalıştırır",
            ),
            (
                "RRF_K",
                RRF_K,
                "İki arama sıralamasını birleştirirken kullanılan yumuşatma sabiti",
            ),
            (
                "TERM_EVIDENCE_MIN_SHORT_ROOT",
                TERM_EVIDENCE_MIN_SHORT_ROOT,
                "Kısa kökler için tamamen kapsanması gereken en kısa uzunluk",
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
            ("CHUNK_SIZE", CHUNK_SIZE, "Özel tokenlar dahil maksimum token"),
            ("CHUNK_OVERLAP", CHUNK_OVERLAP, "Ardışık chunklar arasındaki token tekrarı"),
            (
                "LOCAL_RAG_MODEL",
                MODEL_ALIAS,
                f"Chat modeli; varsayılan {DEFAULT_MODEL_ALIAS}",
            ),
            ("DOCS_DIR", DOCS_DIR, "İndekslenecek doküman klasörü"),
            ("DB_PATH", DB_PATH, "Üretilen SQLite indeks yolu"),
            ("PROJECT_ROOT", PROJECT_PATHS.root, "Aktif Local RAG proje klasörü"),
            (
                "CLI_HISTORY",
                PROJECT_PATHS.history_path,
                "Yerel ve proje bazlı terminal geçmişi",
            ),
            (
                "SESSION_EXPORTS",
                PROJECT_PATHS.session_export_dir,
                "Markdown ve JSON oturum çıktılarının varsayılan klasörü",
            ),
        ],
        footer="Salt okunur görünüm; bu komut ayarları değiştirmez.",
    )


def print_stats():
    stats = get_chunk_stats()
    freshness = get_index_freshness(DOCS_DIR, DB_PATH)
    short_embedding_name = EMBEDDING_MODEL_NAME.rsplit("/", maxsplit=1)[-1]

    print_table(
        "Sistem durumu",
        [("Ayar", "bold"), ("Değer", PRIMARY_BRIGHT)],
        [
            ("Chunk sayısı", stats["total_chunks"]),
            ("Kaynak dosya", stats["source_count"]),
            ("İndeks durumu", freshness.display_status()),
            ("Kaynak filtresi", _source_filter or "kapalı"),
            ("Veritabanı", stats["db_path"]),
            ("Embedding", short_embedding_name),
            ("LLM", MODEL_ALIAS),
            ("Debug", "açık" if DEBUG else "kapalı"),
            ("Top K", TOP_K),
            ("Chunk size / overlap", f"{CHUNK_SIZE} / {CHUNK_OVERLAP}"),
            ("Similarity threshold", SIMILARITY_THRESHOLD),
            ("Context threshold", CONTEXT_SCORE_THRESHOLD),
            ("Context relative margin", CONTEXT_RELATIVE_SCORE_MARGIN),
            ("Hybrid search", "açık" if USE_HYBRID_SEARCH else "kapalı"),
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
            ("Tür", PRIMARY_BRIGHT, "left", True),
            ("Sayfa", None, "right", True),
            ("Chunk", None, "right", True),
        ],
        rows,
        footer=f"Toplam: {len(sources)} dosya, {total_chunks} chunk",
    )


def print_doctor_report():
    checks = run_health_checks()
    print_health_report(checks)


def resolve_indexed_source_name(source_name):
    clean_name = source_name.strip()
    matches = [
        source["source_name"]
        for source in get_indexed_sources()
        if source["source_name"].casefold() == clean_name.casefold()
    ]
    return matches[0] if matches else None


def show_chunk_command(chunk_id):
    try:
        chunk = get_chunk_by_id(chunk_id)
    except Exception as error:
        print_issue(
            "error",
            "Chunk okunamadı.",
            solution="/doctor ile veritabanını kontrol et.",
            error=error,
            debug=DEBUG,
        )
        return False

    if chunk is None:
        print_issue(
            "warning",
            f"ID {chunk_id} için bir chunk bulunamadı.",
            solution="Geçerli ID değerlerini görmek için bir soru sor veya /sources çalıştır.",
        )
        return False

    print_chunk_detail(chunk)
    return True


def set_source_filter(source_name):
    global _source_filter

    if source_name.strip().casefold() in {"off", "none", "kapat", "kapalı"}:
        _source_filter = None
        print_info("Kaynak filtresi kapatıldı.")
        return True

    canonical_name = resolve_indexed_source_name(source_name)
    if canonical_name is None:
        print_issue(
            "warning",
            f"İndekste {source_name} adlı kaynak bulunamadı.",
            solution="Kaynak adlarını görmek için /sources çalıştır.",
        )
        return False

    _source_filter = canonical_name
    print_success(f"Kaynak filtresi etkin · {canonical_name}")
    return True


def print_sources(sources):
    rows = []
    role_labels = {
        "matched": "Eşleşme",
        "neighbor": "Komşu",
    }
    for source in sources:
        rows.append(
            (
                source.source_name,
                source.page_number if source.page_number is not None else "-",
                source.chunk_index if source.chunk_index is not None else "-",
                role_labels.get(source.context_role, source.context_role),
                source.id,
                f"{source.score:.4f}",
            )
        )

    print_table(
        "Kaynaklar",
        [
            ("Dosya", "bold"),
            ("Sayfa", None, "right", True),
            ("Parça", None, "right", True),
            ("Rol", None, "left", True),
            ("ID", "dim", "right", True),
            ("Skor", PRIMARY_BRIGHT, "right", True),
        ],
        rows,
    )


def print_debug_info(question, sources, messages):
    console.rule(f"[bold {DEBUG_COLOR}]DEBUG[/bold {DEBUG_COLOR}]", style=DEBUG_COLOR)
    console.print("[bold]Kullanıcı sorusu[/bold]")
    console.print(question)
    console.print("\n[bold]Retrieved chunks[/bold]")
    for source in sources:
        console.print(
            f"[dim]ID {source.id} · {source.source_name} · "
            f"skor {source.score:.4f}[/dim]"
        )
        console.print(source.chunk_text)

    console.print("\n[bold]Modele gönderilen mesajlar[/bold]")
    for message in messages:
        console.print(f"[bold {DEBUG_COLOR}]{message['role']}[/bold {DEBUG_COLOR}]")
        console.print(message["content"])

    console.rule(style=DEBUG_COLOR)


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
            f"\n[bold {MUTED_AMBER}]{source_name} silinsin mi?[/bold {MUTED_AMBER}] "
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


def print_benchmark_report(report, report_path):
    summary_rows = []
    answer_rows = []

    for model_result in report["models"]:
        if model_result["status"] == "error":
            summary_rows.append((
                model_result["model"],
                "hata",
                f"{model_result['load_seconds']:.3f}",
                "-",
                "-",
                "-",
                "-",
            ))
            answer_rows.append((
                model_result["model"],
                "model yükleme",
                model_result["error"],
            ))
            continue

        summary = model_result["summary"]
        summary_rows.append((
            model_result["model"],
            "kısmi" if model_result["status"] == "partial" else "tamamlandı",
            f"{model_result['load_seconds']:.3f}",
            f"{summary['cold_generation_seconds']:.3f}",
            f"{summary['warm_generation_seconds']:.3f}",
            f"{summary['valid_case_count']}/{summary['case_count']}",
            f"%{summary['average_term_coverage'] * 100:.0f}",
        ))

        for case in model_result["cases"]:
            answer = case["answer"] or case["runs"][-1]["error"] or "cevap yok"

            if len(answer) > 500:
                answer = f"{answer[:497].rstrip()}..."

            answer_rows.append((
                model_result["model"],
                case["question"],
                answer,
            ))

    print_table(
        "Model benchmark",
        [
            ("Model", f"bold {PRIMARY_BRIGHT}", "left", True),
            ("Durum",),
            ("Yükleme", None, "right", True),
            ("İlk yanıt", None, "right", True),
            ("Sıcak yanıt", None, "right", True),
            ("Geçerli", None, "right", True),
            ("Terim", None, "right", True),
        ],
        summary_rows,
    )
    print_table(
        "Benchmark cevapları",
        [
            ("Model", f"bold {PRIMARY_BRIGHT}", "left", True),
            ("Soru", "bold"),
            ("Cevap",),
        ],
        answer_rows,
        footer=f"Ayrıntılı JSON raporu: {report_path}",
    )


def run_benchmark_command(model_aliases=None):
    from app.benchmark import BenchmarkPreparationError, run_model_benchmark

    try:
        with activity("Model benchmark çalıştırılıyor..."):
            report, report_path = run_model_benchmark(
                model_aliases,
                report_path=PROJECT_PATHS.benchmark_report_path,
            )
    except BenchmarkPreparationError as error:
        print_issue(
            "error",
            str(error),
            solution="Önce local-rag reindex ve python eval.py çalıştır.",
        )
        return False
    except Exception as error:
        print_issue(
            "error",
            "Model benchmark tamamlanamadı.",
            solution="/doctor ile modelleri kontrol et; gerekirse /debug on kullan.",
            error=error,
            debug=DEBUG,
        )
        return False

    print_benchmark_report(report, report_path)
    return all(model["status"] == "ok" for model in report["models"])


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

    if leading_command in {"/history", "/repeat", "/export"}:
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
        if command_name == "/history":
            if len(arguments) != 1:
                print_issue("warning", "Kullanım: /history")
            else:
                print_session_history()
            return "handled"

        if command_name == "/repeat":
            if len(arguments) > 2 or (
                len(arguments) == 2 and not arguments[1].isdigit()
            ):
                print_issue("warning", "Kayıt ID hatalı.", solution="Kullanım: /repeat [id]")
            else:
                entry_id = int(arguments[1]) if len(arguments) == 2 else None
                repeat_session_entry(entry_id)
            return "handled"

        if len(arguments) not in {2, 3}:
            print_issue(
                "warning",
                "Dışa aktarım komutu eksik veya hatalı.",
                solution="Kullanım: /export <markdown|json> [dosya-yolu]",
            )
        else:
            output_path = arguments[2] if len(arguments) == 3 else None
            export_session(arguments[1], output_path)
        return "handled"

    if normalized_command in INFO_COMMAND_MESSAGES:
        execute_command(normalized_command)
        return "handled"

    if normalized_command == "/reindex":
        execute_command(normalized_command)
        return "handled"

    if leading_command in {"/show", "/filter", "/ask"}:
        try:
            arguments = shlex.split(stripped_command)
        except ValueError as error:
            print_issue(
                "error",
                "Komut argümanları okunamadı.",
                solution="Boşluk içeren değerleri çift tırnak içine al.",
                error=error,
                debug=DEBUG,
            )
            return "handled"

        command_name = arguments[0].lower()

        if command_name == "/show":
            if len(arguments) != 2 or not arguments[1].isdigit():
                print_issue(
                    "warning",
                    "Chunk ID eksik veya hatalı.",
                    solution="Kullanım: /show <chunk-id>",
                )
            else:
                show_chunk_command(int(arguments[1]))
            return "handled"

        if command_name == "/filter":
            if len(arguments) == 1:
                print_info(f"Kaynak filtresi: {_source_filter or 'kapalı'}")
            elif len(arguments) == 2:
                set_source_filter(arguments[1])
            else:
                print_issue(
                    "warning",
                    "Kaynak filtresi komutu hatalı.",
                    solution="Kullanım: /filter <dosya-adı|off>",
                )
            return "handled"

        source_name = None
        question_parts = arguments[1:]
        if question_parts[:1] == ["--source"]:
            if len(question_parts) < 3:
                print_issue(
                    "warning",
                    "Kaynak veya soru eksik.",
                    solution="Kullanım: /ask --source <dosya-adı> <soru>",
                )
                return "handled"
            source_name = resolve_indexed_source_name(question_parts[1])
            if source_name is None:
                print_issue(
                    "warning",
                    f"İndekste {question_parts[1]} adlı kaynak bulunamadı.",
                    solution="Kaynak adlarını görmek için /sources çalıştır.",
                )
                return "handled"
            question_parts = question_parts[2:]

        if not question_parts:
            print_issue(
                "warning",
                "Soru eksik.",
                solution="Kullanım: /ask [--source <dosya-adı>] <soru>",
            )
            return "handled"

        answer_question(" ".join(question_parts), source_name=source_name)
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

    if leading_command == "/benchmark":
        try:
            arguments = shlex.split(stripped_command)
        except ValueError as error:
            print_issue(
                "error",
                "Model alias'ları okunamadı.",
                solution="Boşluk içeren değerleri çift tırnak içine al.",
                error=error,
                debug=DEBUG,
            )
            return "handled"

        run_benchmark_command(arguments[1:] or [MODEL_ALIAS])
        return "handled"

    if normalized_command.startswith("/"):
        print_issue(
            "warning",
            "Bilinmeyen komut.",
            solution="Komutları görmek için /help yaz.",
        )
        return "handled"

    return None


def answer_question(question, source_name=None, use_active_filter=True):
    if not question.strip():
        print_issue("warning", "Soru boş olamaz.")
        return False

    warn_if_index_is_stale()

    active_source = source_name
    if active_source is None and use_active_filter:
        active_source = _source_filter

    service = RAGService(
        retrieval_func=get_top_chunks,
        llm_factory=get_llm,
    )

    try:
        with rag_progress(MODEL_ALIAS) as progress:
            result = service.answer(
                question,
                source_name=active_source,
                activity_factory=progress.stage,
                context_callback=print_debug_info if DEBUG else None,
                stream_callback=(
                    progress.update_answer if console.is_terminal else None
                ),
            )
    except KeyboardInterrupt:
        print_info("İşlem iptal edildi; kısmi cevap kaydedilmedi.")
        return False
    except EmptyQuestionError:
        print_issue("warning", "Soru boş olamaz.")
        return False
    except EmptyIndexError:
        if active_source:
            print_issue(
                "warning",
                f"{active_source} kaynağında aranabilecek chunk bulunamadı.",
                solution="/sources ile indeksi kontrol et veya /filter off çalıştır.",
            )
            return False
        print_issue(
            "warning",
            "Aranabilecek bir indeks bulunamadı.",
            solution="/reindex çalıştır.",
        )
        return False
    except Exception as error:
        print_issue(
            "error",
            "Dokümanlarda arama yapılamadı.",
            solution="/doctor çalıştır; indeks sorunu varsa /reindex ile yenile.",
            error=error,
            debug=DEBUG,
        )
        return False

    if result.warning:
        print_issue(
            "warning",
            result.warning,
            solution=result.warning_solution,
            error=result.warning_error,
            debug=DEBUG,
        )

    print_answer(result.answer, result.mode, result.best_score)
    if result.sources:
        print_sources(result.sources)
    print_performance(
        result.timings.retrieval_seconds,
        result.timings.generation_seconds,
        result.timings.total_seconds,
    )
    _session_history.add_result(result)
    return True


def main():
    _session_history.clear()
    print_banner()
    input_manager = CLIInputManager(
        PROJECT_PATHS.history_path,
        source_provider=get_indexed_sources,
        fallback_reader=read_prompt,
        echo_handler=print_submitted_prompt,
        status_provider=get_cli_status,
        session_id_provider=get_session_ids,
    )
    input_manager.start()

    try:
        while True:
            try:
                question = input_manager.read().strip()
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
    finally:
        input_manager.save()


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
        "--project",
        metavar="YOL",
        help=(
            "docs/ ve data/ klasörlerini içeren proje kökü "
            f"(ortam alternatifi: {PROJECT_ENV_VAR})."
        ),
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
    ask_parser.add_argument(
        "--source",
        help="Aramayı yalnızca bu indekslenmiş kaynakla sınırlar.",
    )

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

    benchmark_parser = subparsers.add_parser(
        "benchmark",
        help="Modellerin üretim süresi ve cevap kalitesini karşılaştırır.",
    )
    benchmark_parser.add_argument(
        "--models",
        nargs="+",
        default=[MODEL_ALIAS],
        help=f"Karşılaştırılacak model alias'ları (varsayılan: {MODEL_ALIAS})",
    )

    show_parser = subparsers.add_parser(
        "show",
        help="Bir chunk'ın tam metnini ID ile gösterir.",
    )
    show_parser.add_argument("chunk_id", type=int, help="Gösterilecek chunk ID")

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

    parser = build_cli_parser()
    args = parser.parse_args(argv)

    try:
        configure_project(args.project)
    except ProjectConfigurationError as error:
        parser.error(str(error))

    DEBUG = args.debug

    if args.command is None:
        main()
        return 0

    if args.command == "ask":
        question = " ".join(args.question)
        source_name = None
        if args.source:
            source_name = resolve_indexed_source_name(args.source)
            if source_name is None:
                print_issue(
                    "warning",
                    f"İndekste {args.source} adlı kaynak bulunamadı.",
                    solution="Kaynak adlarını görmek için local-rag sources çalıştır.",
                )
                return 1
        if source_name is None:
            return 0 if answer_question(question) else 1
        return 0 if answer_question(question, source_name=source_name) else 1

    if args.command == "add":
        return 0 if add_document_command(args.path) else 1

    if args.command == "remove":
        return 0 if remove_document_command(args.source_name, args.yes) else 1

    if args.command == "benchmark":
        return 0 if run_benchmark_command(args.models) else 1

    if args.command == "show":
        return 0 if show_chunk_command(args.chunk_id) else 1

    return 0 if execute_command(f"/{args.command}") else 1


if __name__ == "__main__":
    raise SystemExit(cli())
