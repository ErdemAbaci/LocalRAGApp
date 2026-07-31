"""Hybrid search ayarlarını gerçek eval seti üzerinde ölçer.

Amaç üç sihirli sayıyı ölçüme bağlamak: hybrid search açık mı olmalı, BM25
`k1`/`b` ne olmalı ve `RRF_K` kaç olmalı. Kıyas ölçütü sıralama metrikleridir
(Recall@k, MRR), çünkü hybrid search'ün çözmek için eklendiği problem
sıralamadır.

Sıra önemli: önce k1/b mevcut RRF_K ile taranır (RRF_K henüz kazananı
etkilemez, çünkü aynı k1/b her RRF_K adayında karşılaştırılacaktır), kazanan
k1/b sabitlenir, sonra RRF_K o k1/b ile taranır. İkisi birlikte taranmaz;
kombinasyon sayısı etiketli vaka sayısına göre gürültüyü büyütür.

Bu araç uygulamanın parçası değildir; ölçüm kaydı olarak repoda tutulur.

Kullanım:

    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python tools/hybrid_search_analysis.py

Kapı davranışını (hard negative sızıntısı) ölçmez; onu `eval.py` yapar. Burada
yalnızca doğru chunk'ın kaçıncı sırada geldiğine bakılır.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import BM25_B, BM25_K1, RRF_K  # noqa: E402
from app.eval_metrics import (  # noqa: E402
    build_case_metrics,
    format_metric,
    summarize_case_metrics,
)
from app.retrieval import get_top_chunks  # noqa: E402

CASES_PATH = "eval_cases.json"
METRIC_TOP_K = 5
METRIC_K_VALUES = (1, 3, 5)
RRF_K_CANDIDATES = (1, 2, 3, 4, 5, 10, 20, 60)
BM25_K1_CANDIDATES = (0.9, 1.2, 1.5, 1.8, 2.0)
BM25_B_CANDIDATES = (0.0, 0.25, 0.5, 0.75, 1.0)


def load_labeled_cases():
    with open(CASES_PATH, "r", encoding="utf-8") as cases_file:
        cases = json.load(cases_file)

    return [case for case in cases if case.get("relevant_chunk_terms")]


def measure(cases, use_hybrid, rrf_k, bm25_k1=BM25_K1, bm25_b=BM25_B):
    case_metrics = []
    ranks = {}

    for case in cases:
        results = get_top_chunks(
            case["question"],
            top_k=METRIC_TOP_K,
            use_hybrid=use_hybrid,
            rrf_k=rrf_k,
            bm25_k1=bm25_k1,
            bm25_b=bm25_b,
        )
        metrics = build_case_metrics(
            results,
            case["relevant_chunk_terms"],
            k_values=METRIC_K_VALUES,
        )
        case_metrics.append(metrics)
        ranks[case["name"]] = metrics["signature_ranks"]

    summary = summarize_case_metrics(case_metrics, k_values=METRIC_K_VALUES)

    return summary, ranks


def print_summary_table(rows):
    header = f"{'ayar':<20}{'R@1':>9}{'R@3':>9}{'R@5':>9}{'MRR':>9}"
    print(header)
    print("-" * len(header))

    for label, summary in rows:
        print(
            f"{label:<20}"
            f"{format_metric(summary['recall_at_1']):>9}"
            f"{format_metric(summary['recall_at_3']):>9}"
            f"{format_metric(summary['recall_at_5']):>9}"
            f"{format_metric(summary['mrr']):>9}"
        )


def print_rank_changes(baseline_ranks, current_ranks, label):
    changed = [
        (name, baseline_ranks[name], current_ranks[name])
        for name in baseline_ranks
        if baseline_ranks[name] != current_ranks[name]
    ]

    print(f"\nDense'e göre sıra değişimi ({label}):")

    if not changed:
        print("  değişim yok")
        return

    for name, before, after in changed:
        print(f"  {name}: {before} -> {after}")


def run_bm25_grid(cases):
    print("=== BM25 k1/b taraması (RRF_K sabit = %s) ===\n" % RRF_K)
    rows = []
    grid_summaries = {}

    for k1 in BM25_K1_CANDIDATES:
        for b in BM25_B_CANDIDATES:
            summary, _ = measure(
                cases, use_hybrid=True, rrf_k=RRF_K, bm25_k1=k1, bm25_b=b
            )
            label = f"k1={k1} b={b}"
            rows.append((label, summary))
            grid_summaries[(k1, b)] = summary

    print_summary_table(rows)

    best = max(
        grid_summaries,
        key=lambda kb: (grid_summaries[kb]["mrr"], grid_summaries[kb]["recall_at_1"]),
    )
    print(f"\nMRR'a göre en iyi (k1, b) = {best}")
    return best, grid_summaries


def run_rrf_grid(cases, bm25_k1, bm25_b):
    print("\n=== RRF_K taraması (BM25 k1=%s b=%s ile) ===\n" % (bm25_k1, bm25_b))
    dense_summary, dense_ranks = measure(
        cases, use_hybrid=False, rrf_k=0, bm25_k1=bm25_k1, bm25_b=bm25_b
    )
    rows = [("dense", dense_summary)]
    hybrid_ranks = {}
    hybrid_summaries = {}

    for rrf_k in RRF_K_CANDIDATES:
        summary, ranks = measure(
            cases, use_hybrid=True, rrf_k=rrf_k, bm25_k1=bm25_k1, bm25_b=bm25_b
        )
        rows.append((f"hybrid k={rrf_k}", summary))
        hybrid_ranks[rrf_k] = ranks
        hybrid_summaries[rrf_k] = summary

    print_summary_table(rows)

    best_k = max(
        RRF_K_CANDIDATES,
        key=lambda k: (hybrid_summaries[k]["mrr"], hybrid_summaries[k]["recall_at_1"]),
    )

    print_rank_changes(dense_ranks, hybrid_ranks[best_k], f"hybrid k={best_k}")
    print(f"\nMRR'a göre en iyi RRF_K = {best_k}")
    return best_k


def main():
    cases = load_labeled_cases()
    print(f"{len(cases)} etiketli vaka ölçülüyor.\n")

    best_kb, _ = run_bm25_grid(cases)
    run_rrf_grid(cases, bm25_k1=best_kb[0], bm25_b=best_kb[1])


if __name__ == "__main__":
    main()
