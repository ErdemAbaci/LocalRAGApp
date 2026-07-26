import unittest

import eval as eval_module


class RelevantCaseEvaluationTests(unittest.TestCase):
    def test_expected_chunk_terms_are_required(self):
        case = {
            "expected_source": "security.txt",
            "min_score": 0.40,
            "expected_chunk_terms": ["gizlilik", "bütünlük", "erişilebilirlik"],
        }
        results = [{
            "source_name": "security.txt",
            "score": 0.75,
            "chunk_text": "Gizlilik ve bütünlük bilgi güvenliği hedefleridir.",
        }]

        passed, detail = eval_module.evaluate_relevant_case(case, results)

        self.assertFalse(passed)
        self.assertIn("erişilebilirlik", detail)

    def test_source_score_and_expected_terms_can_pass_together(self):
        case = {
            "expected_source": "security.txt",
            "min_score": 0.40,
            "expected_chunk_terms": ["gizlilik", "bütünlük", "erişilebilirlik"],
        }
        results = [{
            "source_name": "security.txt",
            "score": 0.75,
            "chunk_text": (
                "Bilgi güvenliği gizlilik, bütünlük ve erişilebilirlik hedeflerini korur."
            ),
        }]

        passed, detail = eval_module.evaluate_relevant_case(case, results)

        self.assertTrue(passed)
        self.assertIn("kavram=3/3", detail)

    def test_expected_context_terms_can_be_satisfied_by_neighbor(self):
        case = {
            "expected_source": "security.txt",
            "min_score": 0.40,
            "expected_context_terms": ["üç kopya", "geri yükleme"],
        }
        neighbor = {
            "id": 2,
            "source_name": "security.txt",
            "source_type": "txt",
            "page_number": None,
            "chunk_index": 2,
            "score": 0.40,
            "chunk_text": "Geri yükleme işlemi düzenli test edilmelidir.",
        }
        results = [{
            "id": 1,
            "source_name": "security.txt",
            "source_type": "txt",
            "page_number": None,
            "chunk_index": 1,
            "score": 0.75,
            "chunk_text": "3-2-1 kuralı üç kopya önerir.",
            "neighbors": [neighbor],
        }]

        passed, detail = eval_module.evaluate_relevant_case(case, results)

        self.assertTrue(passed)
        self.assertIn("context_kavram=2/2", detail)


if __name__ == "__main__":
    unittest.main()
