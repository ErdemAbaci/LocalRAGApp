import unittest

from app.sparse_search import (
    bm25_scores,
    build_document_terms,
    corpus_term_weights,
    inverse_document_frequency,
    term_frequency,
)


class TermFrequencyTests(unittest.TestCase):
    def test_repeated_term_is_counted_each_time(self):
        terms = build_document_terms(["parola parola parola uzunluğu"])[0]

        self.assertEqual(term_frequency("parola", terms, 5), 3)

    def test_turkish_suffix_forms_are_counted(self):
        # "kimlik" sorusu metinde "kimliğin" olarak geçer; birebir eşitlik
        # arayan bir tf bu chunk'ı hiç görmez.
        terms = build_document_terms(["Kimliğin doğrulanması kimlik avını önler."])[0]

        self.assertEqual(term_frequency("kimlik", terms, 5), 2)

    def test_unrelated_term_scores_zero(self):
        terms = build_document_terms(["Yedekleme planı üç kopya önerir."])[0]

        self.assertEqual(term_frequency("parola", terms, 5), 0)


class InverseDocumentFrequencyTests(unittest.TestCase):
    def test_rare_term_outweighs_common_term(self):
        rare = inverse_document_frequency(20, 1)
        common = inverse_document_frequency(20, 18)

        self.assertGreater(rare, common)

    def test_term_in_every_document_stays_positive(self):
        # Klasik idf burada negatife döner ve skoru aşağı çeker; +1 sarmalayıcı
        # bunu engeller.
        self.assertGreater(inverse_document_frequency(10, 10), 0.0)


class CorpusTermWeightTests(unittest.TestCase):
    documents = [
        "Güvenlik hedefleri gizlilik bütünlük erişilebilirlik.",
        "Güvenlik olayı müdahalesi kayıtlarla başlar.",
        "Güvenli yedekleme için 3-2-1 kuralı önerilir.",
    ]

    def weights(self, question):
        return corpus_term_weights(question, build_document_terms(self.documents))

    def test_absent_term_gets_the_highest_weight(self):
        weights = self.weights("Güvenlik duvarı kuralları nedir?")

        self.assertGreater(weights["duvarı"], weights["kuralları"])
        self.assertGreater(weights["kuralları"], weights["güvenlik"])

    def test_weights_cover_exactly_the_question_content_terms(self):
        self.assertEqual(
            set(self.weights("Güvenlik duvarı kuralları nedir?")),
            {"güvenlik", "duvarı", "kuralları"},
        )

    def test_empty_corpus_falls_back_to_equal_weights(self):
        # Ağırlık ölçülemiyorsa eşit saymak, keyfi bir ağırlık uydurmaktan
        # iyidir; kapı da bu durumda eski davranışına döner.
        self.assertEqual(
            corpus_term_weights("Güvenlik duvarı nedir?", []),
            {"güvenlik": 1.0, "duvarı": 1.0},
        )


class BM25Tests(unittest.TestCase):
    documents = [
        "Kimlik avı saldırıları sahte bağlantılarla parola çalmayı hedefler.",
        "Yedekleme planı üç kopyanın iki farklı ortamda tutulmasını önerir.",
        "Güvenlik olayı müdahalesi kayıtların incelenmesiyle başlar.",
    ]

    def score(self, question, documents=None):
        return bm25_scores(
            question,
            build_document_terms(documents or self.documents),
        )

    def test_document_containing_query_terms_scores_highest(self):
        scores = self.score("Kimlik avı nedir?")

        self.assertEqual(max(range(len(scores)), key=lambda i: scores[i]), 0)

    def test_documents_without_query_terms_score_zero(self):
        scores = self.score("Kimlik avı nedir?")

        self.assertEqual(scores[1], 0.0)
        self.assertEqual(scores[2], 0.0)

    def test_rare_term_beats_term_present_everywhere(self):
        documents = [
            "Güvenlik kuralı parola uzunluğunu belirler.",
            "Güvenlik kuralı yedekleme sıklığını belirler.",
            "Güvenlik kuralı erişim yetkisini belirler.",
        ]

        scores = bm25_scores(
            "Güvenlik parola kuralı",
            build_document_terms(documents),
        )

        # "güvenlik" ve "kuralı" üç dokümanda da var, yalnızca "parola" ayırt
        # edici. Sıralamayı ayırt edici terim belirlemeli.
        self.assertGreater(scores[0], scores[1])
        self.assertGreater(scores[0], scores[2])

    def test_term_frequency_saturates(self):
        documents = [
            "parola",
            "parola parola parola parola parola parola parola parola",
        ]

        scores = bm25_scores("parola nedir?", build_document_terms(documents))

        # Sekiz kat tekrar, sekiz kat skor vermemeli.
        self.assertGreater(scores[1], scores[0])
        self.assertLess(scores[1], scores[0] * 8)

    def test_question_with_only_stopwords_scores_zero(self):
        scores = self.score("Bu ne?")

        self.assertEqual(scores, [0.0, 0.0, 0.0])

    def test_empty_corpus_returns_empty_list(self):
        self.assertEqual(bm25_scores("parola", []), [])


if __name__ == "__main__":
    unittest.main()
