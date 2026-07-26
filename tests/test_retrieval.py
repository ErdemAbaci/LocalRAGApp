import unittest

import numpy as np

from app.retrieval import attach_neighbor_chunks, calculate_cosine_similarities


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


if __name__ == "__main__":
    unittest.main()
