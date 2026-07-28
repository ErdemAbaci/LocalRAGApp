import unittest

from app.eval_metrics import (
    build_case_metrics,
    chunk_matches_signature,
    compare_summaries,
    find_signature_ranks,
    find_unmatched_signatures,
    recall_at_k,
    reciprocal_rank,
    summarize_case_metrics,
)


def make_results(*texts):
    return [{"chunk_text": text} for text in texts]


class SignatureMatchingTests(unittest.TestCase):
    def test_all_terms_must_be_present(self):
        chunk = "3-2-1 kuralı üç kopyasının iki farklı ortamda tutulmasını önerir."

        self.assertTrue(
            chunk_matches_signature(chunk, ["üç kopyasının", "iki farklı ortamda"])
        )
        self.assertFalse(
            chunk_matches_signature(chunk, ["üç kopyasının", "çevrimdışı kopya"])
        )

    def test_matching_ignores_case_and_extra_whitespace(self):
        chunk = "Gizlilik,\n  bütünlük   ve erişilebilirlik hedeflerdir."

        self.assertTrue(
            chunk_matches_signature(chunk, ["GİZLİLİK,  bütünlük", "erişilebilirlik"])
        )

    def test_turkish_dotted_capital_i_matches_lowercase(self):
        # "İ".casefold() sonucu "i" + U+0307 olur; ham casefold ile bu eşleşme
        # sessizce başarısız olurdu.
        self.assertTrue(chunk_matches_signature("Gizlilik hedefidir.", ["GİZLİLİK"]))

    def test_turkish_dotless_i_matches_uppercase(self):
        self.assertTrue(chunk_matches_signature("ılık su", ["ILIK"]))

    def test_signature_ranks_use_first_matching_position(self):
        results = make_results(
            "alakasız metin",
            "üç kopyasının iki farklı ortamda tutulması",
            "üç kopyasının iki farklı ortamda tekrar geçtiği metin",
        )

        ranks = find_signature_ranks(
            results,
            [["üç kopyasının", "iki farklı ortamda"]],
        )

        self.assertEqual(ranks, [2])

    def test_missing_signature_rank_is_none(self):
        results = make_results("alakasız metin")

        ranks = find_signature_ranks(results, [["bulunmayan terim"]])

        self.assertEqual(ranks, [None])


class MetricCalculationTests(unittest.TestCase):
    def test_recall_counts_signatures_within_k(self):
        signature_ranks = [1, 4, None]

        self.assertAlmostEqual(recall_at_k(signature_ranks, 1), 1 / 3)
        self.assertAlmostEqual(recall_at_k(signature_ranks, 3), 1 / 3)
        self.assertAlmostEqual(recall_at_k(signature_ranks, 5), 2 / 3)

    def test_recall_without_signatures_is_none(self):
        self.assertIsNone(recall_at_k([], 3))

    def test_reciprocal_rank_uses_best_position(self):
        self.assertEqual(reciprocal_rank([3, 2]), 0.5)
        self.assertEqual(reciprocal_rank([1, None]), 1.0)

    def test_reciprocal_rank_is_zero_when_nothing_found(self):
        self.assertEqual(reciprocal_rank([None, None]), 0.0)

    def test_build_case_metrics_reports_ranks_and_recall(self):
        results = make_results(
            "alakasız",
            "kategorik veriler sayısal değerlere dönüştürülebilir",
        )

        metrics = build_case_metrics(
            results,
            [["kategorik veriler sayısal değerlere dönüştürülebilir"]],
        )

        self.assertEqual(metrics["signature_ranks"], [2])
        self.assertEqual(metrics["recall_at_1"], 0.0)
        self.assertEqual(metrics["recall_at_3"], 1.0)
        self.assertEqual(metrics["reciprocal_rank"], 0.5)

    def test_summary_averages_over_cases(self):
        case_metrics = [
            {"recall_at_1": 1.0, "recall_at_3": 1.0, "recall_at_5": 1.0, "reciprocal_rank": 1.0},
            {"recall_at_1": 0.0, "recall_at_3": 1.0, "recall_at_5": 1.0, "reciprocal_rank": 0.5},
        ]

        summary = summarize_case_metrics(case_metrics)

        self.assertEqual(summary["case_count"], 2)
        self.assertEqual(summary["recall_at_1"], 0.5)
        self.assertEqual(summary["recall_at_3"], 1.0)
        self.assertEqual(summary["mrr"], 0.75)

    def test_summary_without_cases_reports_none(self):
        summary = summarize_case_metrics([])

        self.assertEqual(summary["case_count"], 0)
        self.assertIsNone(summary["mrr"])


class LabelValidationTests(unittest.TestCase):
    def test_unmatched_signature_is_reported(self):
        chunks = [{"chunk_text": "yalnızca bu metin indekste var"}]

        unmatched = find_unmatched_signatures(
            chunks,
            [["bu metin"], ["indekste olmayan imza"]],
        )

        self.assertEqual(unmatched, [["indekste olmayan imza"]])

    def test_all_signatures_matched_returns_empty(self):
        chunks = [
            {"chunk_text": "birinci chunk metni"},
            {"chunk_text": "ikinci chunk metni"},
        ]

        unmatched = find_unmatched_signatures(chunks, [["ikinci chunk"]])

        self.assertEqual(unmatched, [])


class ComparisonTests(unittest.TestCase):
    def test_delta_is_reported_per_metric(self):
        comparisons = compare_summaries(
            {"recall_at_1": 0.5, "recall_at_3": 1.0, "recall_at_5": 1.0, "mrr": 0.75},
            {"recall_at_1": 0.8, "recall_at_3": 1.0, "recall_at_5": 1.0, "mrr": 0.90},
        )

        by_name = {item["name"]: item for item in comparisons}

        self.assertAlmostEqual(by_name["recall_at_1"]["delta"], 0.3)
        self.assertAlmostEqual(by_name["recall_at_3"]["delta"], 0.0)
        self.assertAlmostEqual(by_name["mrr"]["delta"], 0.15)

    def test_missing_baseline_metric_yields_none_delta(self):
        comparisons = compare_summaries({}, {"mrr": 0.9})

        by_name = {item["name"]: item for item in comparisons}

        self.assertIsNone(by_name["mrr"]["delta"])


if __name__ == "__main__":
    unittest.main()
