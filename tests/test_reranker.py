import unittest
from unittest.mock import patch

from app.reranker import RerankerUnavailableError, rerank
from app.retrieval import apply_reranking


def make_result(chunk_id, text, score=0.5):
    return {
        "id": chunk_id,
        "source_name": "notes.txt",
        "source_type": "txt",
        "page_number": None,
        "chunk_index": chunk_id,
        "chunk_text": text,
        "score": score,
        "dense_best_score": 0.9,
    }


def scorer_for(mapping):
    def score_func(_question, texts):
        return [mapping[text] for text in texts]

    return score_func


class RerankTest(unittest.TestCase):
    def test_results_are_reordered_by_cross_encoder_score(self):
        results = [make_result(1, "a"), make_result(2, "b"), make_result(3, "c")]
        reranked = rerank(
            "soru",
            results,
            score_func=scorer_for({"a": 0.1, "b": 0.9, "c": 0.5}),
        )
        self.assertEqual([item["id"] for item in reranked], [2, 3, 1])

    def test_rerank_score_is_attached(self):
        reranked = rerank(
            "soru",
            [make_result(1, "a")],
            score_func=scorer_for({"a": 0.42}),
        )
        self.assertAlmostEqual(reranked[0]["rerank_score"], 0.42)

    def test_ties_keep_the_first_stage_order(self):
        # Cross-encoder iki adaya aynı puanı verirse ilk aşamanın kararı
        # geçerli kalmalı; aksi halde sıralama deterministik olmaz.
        results = [make_result(1, "a"), make_result(2, "b")]
        reranked = rerank(
            "soru",
            results,
            score_func=scorer_for({"a": 0.5, "b": 0.5}),
        )
        self.assertEqual([item["id"] for item in reranked], [1, 2])

    def test_cosine_score_is_not_overwritten(self):
        # Kapı skoru cosine kalmalı. Reranking skoru sıralamaya girer, eşiğe
        # değil; `dense_best_score` ezilirse dört eşik birden kayar.
        results = [make_result(1, "a", score=0.31)]
        reranked = rerank("soru", results, score_func=scorer_for({"a": 9.0}))
        self.assertAlmostEqual(reranked[0]["score"], 0.31)
        self.assertAlmostEqual(reranked[0]["dense_best_score"], 0.9)

    def test_empty_input_returns_empty(self):
        self.assertEqual(rerank("soru", []), [])

    def test_score_count_mismatch_is_an_error(self):
        with self.assertRaises(ValueError):
            rerank("soru", [make_result(1, "a")], score_func=lambda *_: [0.1, 0.2])


class ApplyRerankingTest(unittest.TestCase):
    def setUp(self):
        self.results = [make_result(index, chr(96 + index)) for index in range(1, 6)]

    def test_disabled_reranker_keeps_first_stage_order(self):
        selected = apply_reranking(
            "soru",
            self.results,
            top_k=2,
            use_reranker=False,
        )
        self.assertEqual([item["id"] for item in selected], [1, 2])

    def test_candidate_pool_limits_what_is_rescored(self):
        sizes = []

        def rerank_func(_question, pool):
            sizes.append(len(pool))
            return pool

        apply_reranking(
            "soru",
            self.results,
            top_k=2,
            use_reranker=True,
            candidate_pool=3,
            rerank_func=rerank_func,
        )
        self.assertEqual(sizes, [3])

    def test_pool_is_never_smaller_than_top_k(self):
        # Havuz top_k'dan küçük olursa reranking seçilecek sonuç sayısını kısar
        # ve ölçüm "reranking mi kötüledi, havuz mu daraldı" sorusuna cevap
        # veremez hale gelir.
        sizes = []

        def rerank_func(_question, pool):
            sizes.append(len(pool))
            return pool

        apply_reranking(
            "soru",
            self.results,
            top_k=4,
            use_reranker=True,
            candidate_pool=1,
            rerank_func=rerank_func,
        )
        self.assertEqual(sizes, [4])

    def test_lower_ranked_candidate_can_be_promoted(self):
        # Reranking'in çözmesi beklenen durum: doğru chunk ilk aşamada 2. sırada.
        selected = apply_reranking(
            "soru",
            self.results,
            top_k=1,
            use_reranker=True,
            candidate_pool=5,
            rerank_func=lambda _question, pool: sorted(
                pool,
                key=lambda item: item["id"] != 2,
            ),
        )
        self.assertEqual(selected[0]["id"], 2)

    def test_unavailable_model_falls_back_to_first_stage(self):
        # Model indirilmemişse uygulama çalışmaya devam etmeli; reranking bir
        # iyileştirmedir, ön şart değil.
        with patch(
            "app.reranker.rerank",
            side_effect=RerankerUnavailableError("model yok"),
        ):
            selected = apply_reranking(
                "soru",
                self.results,
                top_k=2,
                use_reranker=True,
            )

        self.assertEqual([item["id"] for item in selected], [1, 2])


if __name__ == "__main__":
    unittest.main()
