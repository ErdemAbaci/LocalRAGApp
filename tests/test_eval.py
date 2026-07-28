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


class KnownGapStatusTests(unittest.TestCase):
    def test_regular_failure_fails_the_gate(self):
        status, gate_result = eval_module.resolve_status({}, False)

        self.assertEqual(status, "FAIL")
        self.assertFalse(gate_result)

    def test_known_gap_failure_is_reported_but_not_gated(self):
        status, gate_result = eval_module.resolve_status({"known_gap": True}, False)

        self.assertEqual(status, "GAP")
        self.assertIsNone(gate_result)

    def test_known_gap_that_starts_passing_is_flagged(self):
        status, gate_result = eval_module.resolve_status({"known_gap": True}, True)

        self.assertEqual(status, "FIXED")
        self.assertTrue(gate_result)

    def test_regular_pass_counts_towards_gate(self):
        status, gate_result = eval_module.resolve_status({}, True)

        self.assertEqual(status, "PASS")
        self.assertTrue(gate_result)


class SignatureRankDescriptionTests(unittest.TestCase):
    def test_missing_rank_is_shown_as_absent(self):
        detail = eval_module.describe_signature_ranks([2, None])

        self.assertEqual(detail, "sıra=2,yok")


if __name__ == "__main__":
    unittest.main()
