import unittest

from app.llm import clean_answer, is_valid_answer


class AnswerCleaningTests(unittest.TestCase):
    def test_parenthesized_piece_citations_are_removed_cleanly(self):
        answer = (
            "Veriler ön işlendikten sonra daha doğru sonuçlar elde edilir (Parça 1). "
            "Algoritmalar daha sonra uygulanır (Parça 3)."
        )

        cleaned = clean_answer(answer)

        self.assertEqual(
            cleaned,
            "Veriler ön işlendikten sonra daha doğru sonuçlar elde edilir. "
            "Algoritmalar daha sonra uygulanır.",
        )

    def test_bracketed_ranges_and_lists_are_removed(self):
        answer = (
            "Cevap: Veri temizleme analiz kalitesini artırır [Parça 1-3]. "
            "Eksik değerler düzeltilir [Parça 1, Parça 2]."
        )

        cleaned = clean_answer(answer)

        self.assertEqual(
            cleaned,
            "Veri temizleme analiz kalitesini artırır. Eksik değerler düzeltilir.",
        )

    def test_cleaned_answer_remains_valid(self):
        answer = (
            "Veri temizleme, hatalı ve eksik verileri düzenleyerek "
            "analize hazırlar (Parça 2)."
        )

        cleaned = clean_answer(answer)

        self.assertTrue(is_valid_answer(cleaned))
        self.assertNotIn("Parça 2", cleaned)


if __name__ == "__main__":
    unittest.main()
