"""Hybrid search ayarlarını gerçek eval seti üzerinde ölçer.

Amaç iki sihirli sayıyı ölçüme bağlamak: hybrid search açık mı olmalı ve `RRF_K`
kaç olmalı. Kıyas ölçütü sıralama metrikleridir (Recall@k, MRR), çünkü hybrid
search'ün çözmek için eklendiği problem sıralamadır.

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

from app.eval_metrics import (  # noqa: E402
    build_case_metrics,
    format_metric,
    summarize_case_metrics,
)
from app.retrieval import get_top_chunks  # noqa: E402

CASES_PATH = "eval_cases.json"
METRIC_TOP_K = 5
METRIC_K_VALUES = (1, 3, 5)
RRF_K_CANDIDATES = (1, 3, 5, 10, 20, 60)


def load_labeled_cases():
    with open(CASES_PATH, "r", encoding="utf-8") as cases_file:
        cases = json.load(cases_file)

    return [case for case in cases if case.get("relevant_chunk_terms")]


def measure(cases, use_hybrid, rrf_k):
    case_metrics = []
    ranks = {}

    for case in cases:
        results = get_top_chunks(
            case["question"],
            top_k=METRIC_TOP_K,
            use_hybrid=use_hybrid,
            rrf_k=rrf_k,
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
    header = f"{'ayar':<16}{'R@1':>9}{'R@3':>9}{'R@5':>9}{'MRR':>9}"
    print(header)
    print("-" * len(header))

    for label, summary in rows:
        print(
            f"{label:<16}"
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


def main():
    cases = load_labeled_cases()
    print(f"{len(cases)} etiketli vaka ölçülüyor.\n")

    dense_summary, dense_ranks = measure(cases, use_hybrid=False, rrf_k=0)
    rows = [("dense", dense_summary)]
    hybrid_ranks = {}

    for rrf_k in RRF_K_CANDIDATES:
        summary, ranks = measure(cases, use_hybrid=True, rrf_k=rrf_k)
        rows.append((f"hybrid k={rrf_k}", summary))
        hybrid_ranks[rrf_k] = ranks

    print_summary_table(rows)

    best_k = max(
        RRF_K_CANDIDATES,
        key=lambda k: (
            dict(rows)[f"hybrid k={k}"]["mrr"],
            dict(rows)[f"hybrid k={k}"]["recall_at_1"],
        ),
    )

    print_rank_changes(dense_ranks, hybrid_ranks[best_k], f"hybrid k={best_k}")
    print(f"\nMRR'a göre en iyi RRF_K = {best_k}")


if __name__ == "__main__":
    main()
