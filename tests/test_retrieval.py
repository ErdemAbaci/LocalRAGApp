import unittest

import numpy as np

from app.retrieval import (
    attach_neighbor_chunks,
    calculate_cosine_similarities,
    gate_score,
    rank_positions,
    reciprocal_rank_fusion,
)


class CosineSimilarityTests(unittest.TestCase):
    def test_normalized_dot_product_ranks_closest_vector_first(self):
        question = np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32)
        chunks = np.asarray([
            [0.9, 0.1, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0],
        ], dtype=np.float32)

        scores = calculate_cosine_similarities(question, chunks)

        self.assertEqual(scores.shape, (3,))
        self.assertGreater(scores[0], scores[1])
        self.assertAlmostEqual(float(scores[1]), 0.0)
        self.assertAlmostEqual(float(scores[2]), 0.0)
        self.assertTrue(np.isfinite(scores).all())

    def test_neighbors_follow_document_order_instead_of_score_order(self):
        ranked = [
            self.make_result(2, page=1, chunk_index=2, score=0.90),
            self.make_result(1, page=1, chunk_index=1, score=0.80),
            self.make_result(4, page=2, chunk_index=2, score=0.70),
            self.make_result(3, page=2, chunk_index=1, score=0.60),
        ]

        enriched = attach_neighbor_chunks(ranked, [ranked[0]], radius=1)

        self.assertEqual(
            [neighbor["id"] for neighbor in enriched[0]["neighbors"]],
            [1, 3],
        )

    @staticmethod
    def make_result(chunk_id, page, chunk_index, score):
        return {
            "id": chunk_id,
            "source_name": "guide.pdf",
            "source_type": "pdf",
            "page_number": page,
            "chunk_index": chunk_index,
            "chunk_text": f"Chunk {chunk_id}",
            "score": score,
        }


class RankPositionTests(unittest.TestCase):
    def test_ranks_start_at_one_and_follow_score_order(self):
        self.assertEqual(rank_positions([0.2, 0.9, 0.5]), [3, 1, 2])

    def test_ties_are_broken_by_index_for_determinism(self):
        self.assertEqual(rank_positions([0.5, 0.5, 0.5]), [1, 2, 3])

    def test_zero_scores_are_unranked_when_only_positive(self):
        # BM25'te sıfır "hiçbir sorgu kelimesi geçmiyor" demektir; bunları
        # sıralamak keyfi eşitlik sırasını sinyal sanmak olur.
        self.assertEqual(
            rank_positions([1.4, 0.0, 0.3, 0.0], only_positive=True),
            [1, None, 2, None],
        )


class ReciprocalRankFusionTests(unittest.TestCase):
    def test_sparse_signal_can_lift_a_lower_ranked_dense_chunk(self):
        # Ölçülen gerçek durum: cevabı içeren chunk cosine'de 4. sırada ama
        # sorunun kelimeleri birebir onda geçiyor.
        dense = [0.60, 0.58, 0.56, 0.54]
        sparse = [0.0, 0.0, 0.0, 8.0]

        fused = reciprocal_rank_fusion(dense, sparse, rrf_k=10)

        self.assertGreater(fused[3], fused[0])

    def test_falls_back_to_dense_order_when_sparse_is_silent(self):
        dense = [0.30, 0.90, 0.60]

        fused = reciprocal_rank_fusion(dense, [0.0, 0.0, 0.0], rrf_k=10)

        self.assertEqual(rank_positions(fused), rank_positions(dense))

    def test_appearing_in_both_lists_beats_appearing_in_one(self):
        # k'dan bağımsız temel mekanizma: iki sinyalin birden gördüğü chunk,
        # yalnızca birinin gördüğü chunk'ın önüne geçer.
        fused = reciprocal_rank_fusion([0.9, 0.8], [0.0, 4.0], rrf_k=1000)

        self.assertGreater(fused[1], fused[0])

    def test_small_rrf_k_rewards_one_strong_rank_over_two_average_ones(self):
        # k'nın asıl işi bu: küçük k tek bir tepe sırayı ödüllendirir, büyük k
        # sıra farklarını düzleştirip istikrarlı ortalamayı öne alır. Hangisinin
        # bizim korpusumuzda doğru olduğu ölçüm sorusudur, gelenek sorusu değil.
        dense = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
        # Tepeci chunk: dense 1., sparse 10. Dengeli chunk: dense 3., sparse 4.
        sparse = [0.1, 1.0, 0.7, 0.9, 0.8, 0.6, 0.5, 0.4, 0.3, 0.2]

        peaked, balanced = 0, 2

        aggressive = reciprocal_rank_fusion(dense, sparse, rrf_k=1)
        flattened = reciprocal_rank_fusion(dense, sparse, rrf_k=60)

        self.assertGreater(aggressive[peaked], aggressive[balanced])
        self.assertLess(flattened[peaked], flattened[balanced])


class GateScoreTests(unittest.TestCase):
    def test_uses_dense_best_score_instead_of_list_order(self):
        results = [
            {"score": 0.42, "dense_best_score": 0.71},
            {"score": 0.71, "dense_best_score": 0.71},
        ]

        self.assertAlmostEqual(gate_score(results), 0.71)

    def test_falls_back_to_max_score_when_key_is_absent(self):
        self.assertAlmostEqual(gate_score([{"score": 0.31}, {"score": 0.55}]), 0.55)

    def test_empty_results_score_zero(self):
        self.assertEqual(gate_score([]), 0.0)


if __name__ == "__main__":
    unittest.main()
