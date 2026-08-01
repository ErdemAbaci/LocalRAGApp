"""Cross-encoder yeniden sıralamasını gerçek eval seti üzerinde ölçer.

İki soruya birden cevap arar, çünkü reranking'in tek başına "daha isabetli"
olması yeterli değildir:

1. **Kazanç.** Doğru chunk daha sık 1. sıraya geliyor mu? Ölçüt sıralama
   metrikleridir (Recall@k, MRR); reranking'in çözmek için eklendiği problem
   sıralamadır.
2. **Bedel.** Her soruya kaç saniye ekliyor? Cross-encoder önceden
   hesaplanamaz, yani bu maliyet her soruda yeniden ödenir.

Aday havuzu da taranır. Havuzu büyütmek ilk aşamada bedavadır (korpusun tamamı
zaten skorlanıyor), pahalı olan ikinci aşamadır ve süre aday sayısıyla doğrusal
artar. Aranan şey kazancın düzleştiği noktadır.

İlk çağrı model yüklemesini içerir ve yanıltıcı biçimde yavaştır; bu yüzden
ölçümden önce bir ısınma turu yapılır.

Bu araç uygulamanın parçası değildir; ölçüm kaydı olarak repoda tutulur.

Kullanım:

    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python tools/reranker_analysis.py
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python tools/reranker_analysis.py BAAI/bge-reranker-v2-m3

Model adı verilirse `app/config.RERANKER_MODEL` yerine o kullanılır. Bu, bir
negatif sonucu tek modele dayandırmamak için gerekli: "reranking bu korpusta
işe yaramıyor" ile "bu model Türkçede zayıf" farklı iddialardır ve ancak ikinci
bir modelle ayrılabilir.

Kapı davranışını (hard negative sızıntısı) ölçmez; onu `eval.py` yapar.
"""

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import (  # noqa: E402
    RERANK_CANDIDATE_POOL,
    RERANK_MAX_LENGTH,
    RERANKER_MODEL,
)
from app.eval_metrics import (  # noqa: E402
    build_case_metrics,
    format_metric,
    summarize_case_metrics,
)
from app.reranker import rerank, score_pairs  # noqa: E402
from app.retrieval import get_top_chunks  # noqa: E402

CASES_PATH = "eval_cases.json"
METRIC_TOP_K = 5
METRIC_K_VALUES = (1, 3, 5)
POOL_CANDIDATES = (5, 10, 15, 20, 30)


def load_labeled_cases():
    with open(CASES_PATH, "r", encoding="utf-8") as cases_file:
        cases = json.load(cases_file)

    return [case for case in cases if case.get("relevant_chunk_terms")]


def build_rerank_func(model_name):
    """Verilen modeli bir kez yükleyip `rerank_func` olarak paketler.

    Ölçüm aracı modeli doğrudan yükler; `app/reranker.py` içindeki süreç
    genelindeki tekil örnek tek bir modele bağlıdır ve iki modeli aynı
    çalıştırmada karşılaştırmayı imkânsız kılardı.
    """
    from sentence_transformers import CrossEncoder

    encoder = CrossEncoder(model_name, max_length=RERANK_MAX_LENGTH)

    def rerank_func(question, pool):
        return rerank(
            question,
            pool,
            score_func=lambda q, texts: score_pairs(q, texts, model=encoder),
        )

    return rerank_func


def measure(cases, use_reranker, candidate_pool=RERANK_CANDIDATE_POOL, rerank_func=None):
    case_metrics = []
    ranks = {}
    elapsed = 0.0

    for case in cases:
        start = time.perf_counter()
        results = get_top_chunks(
            case["question"],
            top_k=METRIC_TOP_K,
            use_reranker=use_reranker,
            candidate_pool=candidate_pool,
            rerank_func=rerank_func,
        )
        elapsed += time.perf_counter() - start

        metrics = build_case_metrics(
            results,
            case["relevant_chunk_terms"],
            k_values=METRIC_K_VALUES,
        )
        case_metrics.append(metrics)
        ranks[case["name"]] = metrics["signature_ranks"]

    summary = summarize_case_metrics(case_metrics, k_values=METRIC_K_VALUES)
    summary["seconds_per_question"] = elapsed / len(cases)

    return summary, ranks


def print_summary_table(rows):
    header = f"{'ayar':<22}{'R@1':>9}{'R@3':>9}{'R@5':>9}{'MRR':>9}{'sn/soru':>10}"
    print(header)
    print("-" * len(header))

    for label, summary in rows:
        print(
            f"{label:<22}"
            f"{format_metric(summary['recall_at_1']):>9}"
            f"{format_metric(summary['recall_at_3']):>9}"
            f"{format_metric(summary['recall_at_5']):>9}"
            f"{format_metric(summary['mrr']):>9}"
            f"{summary['seconds_per_question']:>10.3f}"
        )


def print_rank_changes(baseline_ranks, current_ranks):
    improved = []
    worsened = []

    for name, before in baseline_ranks.items():
        after = current_ranks[name]

        if before == after:
            continue

        first_before = min((rank for rank in before if rank), default=None)
        first_after = min((rank for rank in after if rank), default=None)

        entry = (name, before, after)

        if first_after is None:
            worsened.append(entry)
        elif first_before is None or first_after < first_before:
            improved.append(entry)
        elif first_after > first_before:
            worsened.append(entry)

    print("\nSıra değişimi (hybrid -> hybrid + reranking):")

    if not improved and not worsened:
        print("  değişim yok")
        return

    for name, before, after in improved:
        print(f"  IYILESTI  {name}: {before} -> {after}")

    for name, before, after in worsened:
        print(f"  KOTULESTI {name}: {before} -> {after}")


def main():
    model_name = sys.argv[1] if len(sys.argv) > 1 else RERANKER_MODEL
    cases = load_labeled_cases()
    print(f"{len(cases)} etiketli vaka | model: {model_name}\n")

    rerank_func = build_rerank_func(model_name)

    # Isınma: ilk çağrı ilk batch'in maliyetini içerir ve süre ölçümünü bozar.
    get_top_chunks(
        cases[0]["question"],
        top_k=METRIC_TOP_K,
        use_reranker=True,
        rerank_func=rerank_func,
    )

    baseline_summary, baseline_ranks = measure(cases, use_reranker=False)
    rows = [("reranking kapalı", baseline_summary)]
    pool_ranks = {}

    for pool in POOL_CANDIDATES:
        summary, ranks = measure(
            cases,
            use_reranker=True,
            candidate_pool=pool,
            rerank_func=rerank_func,
        )
        rows.append((f"havuz={pool}", summary))
        pool_ranks[pool] = ranks

    print_summary_table(rows)
    print_rank_changes(baseline_ranks, pool_ranks[RERANK_CANDIDATE_POOL])


if __name__ == "__main__":
    main()
