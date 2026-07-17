import unittest

import numpy as np

from app.retrieval import calculate_cosine_similarities


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


if __name__ == "__main__":
    unittest.main()
