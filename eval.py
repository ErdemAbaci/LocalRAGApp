import argparse
import json
import math
from pathlib import Path

from app.config import NO_EVIDENCE_ANSWER, SIMILARITY_THRESHOLD, TOP_K
from app.database import get_all_chunks, get_chunk_stats
from app.eval_metrics import (
    build_case_metrics,
    compare_summaries,
    find_unmatched_signatures,
    format_metric,
    normalize_text,
    summarize_case_metrics,
)
from app.llm import is_valid_answer
from app.rag_service import RAGService
from app.retrieval import gate_score, get_top_chunks
from app.term_evidence import term_coverage


EVAL_CASES_PATH = Path(__file__).with_name("eval_cases.json")
BASELINE_PATH = Path(__file__).with_name("eval_baseline.json")
EXPECTED_EMBEDDING_DIMENSION = 384

# Metrikler pass/fail kapısından daha geniş bir pencereye bakar. Böylece doğru
# chunk TOP_K dışına düştüğünde "hiç bulunamadı" yerine kaçıncı sırada olduğu
# görülür. Karar mantığı yine TOP_K ile çalışır.
METRIC_TOP_K = 5
METRIC_K_VALUES = (1, 3, 5)

ANSWER_QUALITY_CASES = [
    ("", False),
    ("Kısa cevap", False),
    ("Kaynak: [Parça 1-3]", False),
    # Modelin reddi artık geçerli bir cevaptır ve `rag_service` onu no_evidence
    # olarak ele alır. Eskiden `false_no_evidence` sayılıp kaynak metinle
    # değiştiriliyordu; ölçümde bu, modelin DOĞRU reddini siliyordu.
    (NO_EVIDENCE_ANSWER, True),
    ("Gönderinin " * 18, False),
    ("Veri madenciliği, verilerden anlamlı bilgi çıkarma sürecidir.", True),
]


def load_eval_cases():
    return json.loads(EVAL_CASES_PATH.read_text(encoding="utf-8"))


def validate_index():
    chunks = get_all_chunks()

    if not chunks:
        return False, "Index boş. Önce /reindex çalıştır."

    for chunk in chunks:
        embedding = chunk["embedding"]

        if len(embedding) != EXPECTED_EMBEDDING_DIMENSION:
            return False, (
                f"chunk_id={chunk['id']} embedding boyutu {len(embedding)}; "
                f"beklenen {EXPECTED_EMBEDDING_DIMENSION}."
            )

        if not all(math.isfinite(value) for value in embedding):
            return False, f"chunk_id={chunk['id']} geçersiz embedding değeri içeriyor."

    return True, f"{len(chunks)} chunk ve embedding değerleri sağlıklı."


def validate_answer_quality():
    for answer, expected in ANSWER_QUALITY_CASES:
        actual = is_valid_answer(answer)

        if actual != expected:
            return False, f"beklenen={expected}, gelen={actual}, cevap={answer!r}"

    return True, f"{len(ANSWER_QUALITY_CASES)} cevap kalite kontrolü başarılı."


def validate_case_labels(cases):
    """İmzaların indekste gerçekten karşılığı olduğunu doğrular.

    İmza bazlı etiketlemenin tek gerçek riski, yanlış yazılmış bir imzanın
    sessizce "bulunamadı" sayılıp metrikleri haksız yere düşürmesidir. Bu
    kontrol bozuk etiketi ayrı bir hata olarak yüzeye çıkarır.
    """
    chunks = get_all_chunks()
    broken_labels = []

    for case in cases:
        signatures = case.get("relevant_chunk_terms", [])

        if not signatures:
            continue

        unmatched = find_unmatched_signatures(chunks, signatures)

        for signature in unmatched:
            broken_labels.append(f"{case['name']}: {signature}")

    if broken_labels:
        return False, "indekste karşılığı olmayan imzalar: " + "; ".join(broken_labels)

    labeled_count = sum(
        len(case.get("relevant_chunk_terms", []))
        for case in cases
    )

    return True, f"{labeled_count} imzanın tamamı indekste bulundu."


def evaluate_relevant_case(case, results):
    if not results:
        return False, "Retrieval sonucu gelmedi."

    best_result = results[0]
    expected_source = case["expected_source"]
    min_score = case.get("min_score", SIMILARITY_THRESHOLD)

    if best_result["source_name"] != expected_source:
        return False, (
            f"beklenen kaynak={expected_source}, "
            f"gelen={best_result['source_name']}"
        )

    # Kapı skoru kullanılır, listenin ilk elemanının skoru değil. Hybrid
    # sıralamada birinci sıradaki chunk düşük cosine alabilir ("Yedekleme neden
    # gereklidir?" sorusunda doğru chunk 0.1972 alıyor); o sayıyı eşikle
    # karşılaştırmak doğru sonucu başarısız gösterir. min_score'un asıl sorduğu
    # şey "bu soru kapsam içinde mi", uygulamanın kendi kapısıyla aynı sorudur.
    gate = gate_score(results)

    if gate < min_score:
        return False, f"kapı skoru={gate:.4f}, minimum={min_score:.4f}"

    expected_chunk_terms = case.get("expected_chunk_terms", [])
    normalized_chunk = normalize_text(best_result["chunk_text"])
    missing_terms = [
        term
        for term in expected_chunk_terms
        if normalize_text(term) not in normalized_chunk
    ]

    if missing_terms:
        return False, f"en iyi chunk içinde eksik kavramlar: {', '.join(missing_terms)}"

    # Kelime kanıtı kapısı alakalı vakalarda da kontrol edilir. Aksi halde
    # retrieval doğru parçayı bulup kapı onu reddettiğinde vaka PASS görünür ve
    # kullanıcı "bu bilgi dokümanlarda yok" cevabı alır. Ölçümde tam olarak bu
    # oldu (`stub_vs_mock`). LLM yüklenmez; yalnızca kapı çalıştırılır.
    service = RAGService()
    matched_chunks = service.select_matched_context_chunks(results)
    context_chunks = service.order_context_chunks(
        service.expand_context_chunks(matched_chunks),
        matched_chunks,
    )
    term_weights = results[0].get("question_term_weights")

    if not service.has_term_evidence(case["question"], context_chunks, term_weights):
        coverage = term_coverage(
            case["question"],
            context_chunks,
            weights=term_weights,
        )
        return False, (
            "kelime kanıtı kapısı reddetti: "
            f"kapsama={format_metric(coverage)}, "
            f"eşik={service.term_evidence_threshold:.2f}"
        )

    expected_context_terms = case.get("expected_context_terms", [])
    if expected_context_terms:
        normalized_context = normalize_text("\n".join(
            chunk["chunk_text"]
            for chunk in context_chunks
        ))
        missing_context_terms = [
            term
            for term in expected_context_terms
            if normalize_text(term) not in normalized_context
        ]

        if missing_context_terms:
            return False, (
                "seçilen context içinde eksik kavramlar: "
                f"{', '.join(missing_context_terms)}"
            )

    detail = (
        f"kaynak={best_result['source_name']}, "
        f"skor={best_result['score']:.4f}"
    )

    if expected_chunk_terms:
        detail += f", kavram={len(expected_chunk_terms)}/{len(expected_chunk_terms)}"
    if expected_context_terms:
        detail += (
            f", context_kavram="
            f"{len(expected_context_terms)}/{len(expected_context_terms)}"
        )

    return True, detail


class RefusingLLM:
    """Context'te cevap bulamadığını doğru biçimde söyleyen model."""

    def generate_answer(self, _messages):
        return NO_EVIDENCE_ANSWER


class FabricatingLLM:
    """Context'le ilgisi olmayan bir cevap uyduran model."""

    def generate_answer(self, _messages):
        return (
            "Çikolatalı kek fırında pişirilir ve hamura kabartma tozu "
            "eklenmelidir. Kekin üzerine pudra şekeri serpilir."
        )


class FailingLLM:
    """Geçersiz üretim yapan model; akış `fallback_extractive`e düşer.

    Bu dal manuel testte sızdırdı ve o zamana kadar hiç sınanmamıştı: model
    bozuk bir cevap üretti, sistem kaynak metnine döndü ve alakasız bir cümleyi
    cevap olarak gösterdi. Groundedness bu yolu koruyamaz, çünkü metin zaten
    context'ten gelir ve tanım gereği dayanaklıdır.
    """

    def generate_answer(self, _messages):
        return "Kısa."


def evaluate_answer_mode(case):
    """Kullanıcının gerçekte aldığı kararı doğrular.

    Eskiden bu kontrol LLM yüklenirse hata verirdi; kelime kanıtı kapısı
    kanıtsız soruyu modele hiç göndermiyordu. Kapı alan filtresine
    indirildikten sonra o varsayım geçersiz: hard negative sorular artık
    **kasıtlı olarak** modele ulaşır ve karar cevaba bakılarak verilir.

    Bu, kararın bir kısmını deterministik olmaktan çıkarır — gerçek modelin
    reddedip reddetmeyeceğini eval ölçemez. Ölçebileceği şey bizim tarafımızın
    sözleşmesidir ve iki dalı da burada sınanır:

    - Model doğru davranıp reddederse bu red NİHAİ olmalı. Eski `false_no_evidence`
      koruması tam burada modelin doğru reddini siliyordu.
    - Model uydurursa groundedness kapısı cevabı `ungrounded` ile kesmeli.

    Gerçek modelin hangi dala gireceği manuel testin konusudur; AGENTS.md
    bölüm 9'daki soru listesi bunun içindir.
    """
    expected_mode = case["expect_answer_mode"]

    refused = RAGService(llm_factory=RefusingLLM).answer(case["question"])

    if refused.mode != expected_mode:
        return False, (
            f"model reddettiğinde mod={refused.mode}, beklenen={expected_mode}"
        )

    if refused.sources:
        return False, "reddedilen cevapta kaynak gösterildi"

    fabricated = RAGService(llm_factory=FabricatingLLM).answer(case["question"])

    if fabricated.mode not in ("no_evidence", "ungrounded"):
        return False, (
            f"model uydurduğunda mod={fabricated.mode}, "
            "beklenen no_evidence veya ungrounded"
        )

    failed = RAGService(llm_factory=FailingLLM).answer(case["question"])

    if failed.mode != "no_evidence":
        return False, (
            f"model bozuk üretince mod={failed.mode}, beklenen no_evidence"
        )

    return True, (
        f"red -> {refused.mode}, uydurma -> {fabricated.mode}, "
        f"bozuk -> {failed.mode}"
    )


def evaluate_not_found_case(case, results):
    if "expect_answer_mode" in case:
        passed, detail = evaluate_answer_mode(case)

        if results:
            detail += f", skor={gate_score(results):.4f}"

        return passed, detail

    if not results:
        return True, "Sonuç yok; beklenen davranış."

    # Hybrid sıralamada results[0] en yüksek cosine olmayabilir; buradan okumak
    # hard negative kapısını sessizce gevşetir.
    best_score = gate_score(results)
    max_score = case.get("max_score", SIMILARITY_THRESHOLD)

    if best_score >= max_score:
        return False, f"skor={best_score:.4f}, maksimum={max_score:.4f}"

    return True, f"skor={best_score:.4f}, eşik altında"


def describe_signature_ranks(signature_ranks):
    positions = [
        "yok" if rank is None else str(rank)
        for rank in signature_ranks
    ]

    return "sıra=" + ",".join(positions)


def evaluate_case(case):
    # Tek retrieval yapılır. Sonuçlar skora göre sıralı olduğu için ilk TOP_K
    # eleman, top_k=TOP_K ile yapılmış bir aramanın sonucuyla aynıdır.
    results = get_top_chunks(case["question"], top_k=METRIC_TOP_K)
    gate_results = results[:TOP_K]
    expectation = case["expectation"]

    if expectation == "relevant":
        passed, detail = evaluate_relevant_case(case, gate_results)
    elif expectation == "not_found":
        passed, detail = evaluate_not_found_case(case, gate_results)
    else:
        return False, f"Bilinmeyen expectation: {expectation}", None

    metrics = None
    signatures = case.get("relevant_chunk_terms", [])

    if signatures:
        metrics = build_case_metrics(results, signatures, k_values=METRIC_K_VALUES)
        detail += ", " + describe_signature_ranks(metrics["signature_ranks"])

    if expectation == "not_found" and results:
        metrics = {"best_score": gate_score(results)}

    return passed, detail, metrics


def resolve_status(case, passed):
    """Bilinen boşlukları gerçek regression'dan ayırır.

    Hard negative vakaların bir kısmının mevcut eşiklerle başarısız olması
    beklenir. Bunları FAIL saymak eval'i kalıcı kırmızıya çevirir ve kısa
    sürede görmezden gelinmesine yol açar. `known_gap` işaretli vakalar
    raporlanır ve ölçülür ama pass/fail kapısını düşürmez.
    """
    known_gap = case.get("known_gap", False)

    if passed and known_gap:
        return "FIXED", True

    if passed:
        return "PASS", True

    if known_gap:
        return "GAP", None

    return "FAIL", False


def build_report(cases):
    relevant_metrics = []
    hard_negative_scores = {}
    case_rows = []
    gate_passed = 0
    gate_total = 0
    fixed_cases = []

    for case in cases:
        passed, detail, metrics = evaluate_case(case)
        status, gate_result = resolve_status(case, passed)

        case_rows.append((status, case["name"], detail))

        if gate_result is not None:
            gate_total += 1
            if gate_result:
                gate_passed += 1

        if status == "FIXED":
            fixed_cases.append(case["name"])

        if metrics and "signature_ranks" in metrics:
            relevant_metrics.append(metrics)

        if case.get("difficulty") == "hard" and metrics and "best_score" in metrics:
            hard_negative_scores[case["name"]] = metrics["best_score"]

    summary = summarize_case_metrics(relevant_metrics, k_values=METRIC_K_VALUES)

    return {
        "case_rows": case_rows,
        "summary": summary,
        "hard_negative_scores": hard_negative_scores,
        "gate_passed": gate_passed,
        "gate_total": gate_total,
        "fixed_cases": fixed_cases,
    }


def print_metrics(summary, hard_negative_scores):
    print("\nRetrieval metrikleri")
    print(f"  Etiketli vaka        : {summary['case_count']}")

    for k in METRIC_K_VALUES:
        value = format_metric(summary.get(f"recall_at_{k}"))
        print(f"  Recall@{k}             : {value}")

    print(f"  MRR                  : {format_metric(summary.get('mrr'))}")

    if hard_negative_scores:
        print("\nHard negative skorları (eşik: "
              f"{SIMILARITY_THRESHOLD:.2f})")

        for name, score in sorted(
            hard_negative_scores.items(),
            key=lambda item: item[1],
            reverse=True,
        ):
            marker = "üstünde" if score >= SIMILARITY_THRESHOLD else "altında"
            print(f"  {score:.4f}  {marker:<8} {name}")


def load_baseline():
    if not BASELINE_PATH.exists():
        return None

    return json.loads(BASELINE_PATH.read_text(encoding="utf-8"))


def write_baseline(report):
    payload = {
        "summary": report["summary"],
        "hard_negative_scores": report["hard_negative_scores"],
    }
    BASELINE_PATH.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def print_comparison(report):
    baseline = load_baseline()

    if baseline is None:
        print(
            f"\nBaseline bulunamadı ({BASELINE_PATH.name}). "
            "Oluşturmak için: python eval.py --update-baseline"
        )
        return

    print("\nBaseline karşılaştırması")
    comparisons = compare_summaries(
        baseline.get("summary", {}),
        report["summary"],
        k_values=METRIC_K_VALUES,
    )

    for comparison in comparisons:
        delta = comparison["delta"]

        if delta is None:
            delta_text = "-"
        elif delta > 0:
            delta_text = f"+{delta:.4f} iyileşme"
        elif delta < 0:
            delta_text = f"{delta:.4f} gerileme"
        else:
            delta_text = "değişmedi"

        print(
            f"  {comparison['name']:<12} "
            f"{format_metric(comparison['baseline'])} -> "
            f"{format_metric(comparison['current'])}  {delta_text}"
        )

    baseline_scores = baseline.get("hard_negative_scores", {})
    current_scores = report["hard_negative_scores"]
    changed = [
        (name, baseline_scores[name], current_scores[name])
        for name in sorted(current_scores)
        if name in baseline_scores
        and abs(current_scores[name] - baseline_scores[name]) >= 0.0001
    ]

    if changed:
        print("\n  Hard negative skor değişimi")
        for name, old_score, new_score in changed:
            direction = "düştü" if new_score < old_score else "yükseldi"
            print(f"    {name}: {old_score:.4f} -> {new_score:.4f} ({direction})")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Local RAG retrieval ve indeks değerlendirmesi.",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Sonuçları eval_baseline.json ile karşılaştırır.",
    )
    parser.add_argument(
        "--update-baseline",
        action="store_true",
        help="Güncel metrikleri baseline olarak kaydeder.",
    )

    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    stats = get_chunk_stats()

    if stats["total_chunks"] == 0:
        print("Index boş. Önce uygulamada /reindex çalıştır.")
        return 1

    cases = load_eval_cases()

    checks = [
        ("index_health", validate_index()),
        ("answer_quality", validate_answer_quality()),
        ("case_labels", validate_case_labels(cases)),
    ]

    passed_count = 0

    for name, (check_passed, detail) in checks:
        status = "PASS" if check_passed else "FAIL"
        print(f"{status:<5} {name} - {detail}")

        if check_passed:
            passed_count += 1

    report = build_report(cases)

    for status, name, detail in report["case_rows"]:
        print(f"{status:<5} {name} - {detail}")

    passed_count += report["gate_passed"]
    total_count = report["gate_total"] + len(checks)

    print_metrics(report["summary"], report["hard_negative_scores"])

    if args.compare:
        print_comparison(report)

    print(f"\n{passed_count}/{total_count} test başarılı")

    gap_count = sum(1 for status, _, _ in report["case_rows"] if status == "GAP")
    if gap_count:
        print(
            f"{gap_count} bilinen boşluk (GAP) raporlandı; "
            "pass/fail kapısına dahil değil."
        )

    if report["fixed_cases"]:
        print(
            "Bu vakalar artık geçiyor, known_gap kaldırılabilir: "
            + ", ".join(report["fixed_cases"])
        )

    if args.update_baseline:
        write_baseline(report)
        print(f"\nBaseline güncellendi: {BASELINE_PATH.name}")

    return 0 if passed_count == total_count else 1


if __name__ == "__main__":
    raise SystemExit(main())
