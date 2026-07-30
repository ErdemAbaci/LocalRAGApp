import unittest

from app.term_evidence import (
    QUESTION_STOPWORDS,
    extract_question_terms,
    has_term_evidence,
    normalize_text,
    term_coverage,
    terms_match,
)


def make_chunks(*texts):
    return [{"chunk_text": text} for text in texts]


class NormalizationTests(unittest.TestCase):
    def test_turkish_dotted_capital_i_becomes_plain_i(self):
        # Ham casefold "i" + U+0307 üretir ve eşleşmeyi sessizce bozar.
        self.assertEqual(normalize_text("GİZLİLİK"), "gizlilik")

    def test_turkish_dotless_i_maps_from_capital_i(self):
        self.assertEqual(normalize_text("ILIK"), "ılık")

    def test_whitespace_is_collapsed(self):
        self.assertEqual(normalize_text("  iki   satır\nvar "), "iki satır var")


class StopwordListTests(unittest.TestCase):
    def test_stopwords_use_turkish_characters(self):
        # ASCII yazılmış bir stopword normalize edilmiş metinle hiç eşleşmez ve
        # listeyi sessizce etkisiz bırakır.
        for word in QUESTION_STOPWORDS:
            self.assertEqual(
                word,
                normalize_text(word),
                msg=f"{word!r} normalize edilmiş biçimiyle aynı değil",
            )

    def test_question_words_are_dropped(self):
        terms = extract_question_terms("Yedekleme ne sıklıkla alınmalıdır?")

        self.assertEqual(terms, ["yedekleme", "sıklıkla"])

    def test_short_tokens_are_dropped(self):
        self.assertEqual(extract_question_terms("AB cd efg"), ["efg"])

    def test_repeated_terms_are_counted_once(self):
        terms = extract_question_terms("Veri veri veri madenciliği")

        self.assertEqual(terms, ["veri", "madenciliği"])

    def test_question_with_only_stopwords_yields_no_terms(self):
        self.assertEqual(extract_question_terms("Bu ne?"), [])


class PrefixMatchingTests(unittest.TestCase):
    def test_identical_terms_match(self):
        self.assertTrue(terms_match("parola", "parola"))

    def test_turkish_suffix_is_matched_by_prefix(self):
        self.assertTrue(terms_match("bağlantılar", "bağlantı"))
        self.assertTrue(terms_match("aşamasında", "aşama"))
        self.assertTrue(terms_match("yazılımının", "yazılım"))

    def test_turkish_consonant_mutation_is_matched(self):
        # Ünlüyle başlayan ek gelince sondaki p/ç/t/k yumuşar; harf harf
        # karşılaştırma bunu tolere etmezse kök ortada kopar.
        self.assertTrue(terms_match("süreç", "süreci"))
        self.assertTrue(terms_match("kitap", "kitabı"))

    def test_shared_root_matches_even_when_neither_is_a_prefix(self):
        # "korunulur" ve "korunmak" aynı kökten türer ama hiçbiri diğerinin
        # öneki değildir. Önek kuralı bunu kaçırıyor ve "Kimlik avından nasıl
        # korunulur?" sorusu haksız yere reddediliyordu.
        self.assertTrue(terms_match("korunulur", "korunmak"))

    def test_unrelated_words_do_not_match(self):
        self.assertFalse(terms_match("parola", "paralel"))
        self.assertFalse(terms_match("yedekleme", "yetkili"))

    def test_short_root_matches_when_fully_consumed(self):
        # Kök minimum ortak önekten kısa olduğunda şart, kökün tamamen
        # kapsanmasıdır. Ölçüm bunu zorunlu kıldı: "avı" 3 karakter olduğu için
        # "avından" korpusta hiçbir şeyle eşleşmiyor ve haksız yere en yüksek
        # IDF ağırlığını alıyordu.
        self.assertTrue(terms_match("avından", "avı"))
        self.assertTrue(terms_match("küme", "kümeleme"))

    def test_short_root_below_minimum_still_needs_exact_match(self):
        # İki karakterlik bir önek sinyal değil gürültüdür.
        self.assertFalse(terms_match("av", "avından"))

    def test_diverging_short_words_do_not_match(self):
        self.assertFalse(terms_match("küme", "kumaş"))
        self.assertFalse(terms_match("veri", "vergi"))

    def test_short_terms_still_match_exactly(self):
        # "rag" minimum kökten kısa; tam eşleşme yolu olmasaydı hiç
        # eşleşemez ve rag_definition vakası çökerdi.
        self.assertTrue(terms_match("rag", "rag"))

    def test_derivational_suffixes_sharing_a_root_match(self):
        # Ölçülen ödünleşme: ortak kök kuralı "sayısı" ile "sayısal"ı da
        # eşleştirir. İkisi de "sayı" kökünden gelir; anlam farkı türetme
        # ekindedir ve morfoloji bunu ayıramaz. 19 vakalık ölçümde ortak kök 5,
        # ayrımı en iyi koruyan seçenek çıktı.
        self.assertTrue(terms_match("sayısı", "sayısal"))
        self.assertTrue(terms_match("yapılandırılmalıdır", "yapılır"))

    def test_min_prefix_is_configurable(self):
        self.assertTrue(terms_match("küme", "kümeleme", min_prefix=4))


class CoverageTests(unittest.TestCase):
    def test_full_coverage(self):
        chunks = make_chunks(
            "3-2-1 kuralı üç kopyasının iki farklı ortamda tutulmasını önerir."
        )

        self.assertEqual(term_coverage("3-2-1 kuralı nedir?", chunks), 1.0)

    def test_partial_coverage(self):
        chunks = make_chunks("Güvenlik olayı müdahalesi anlatılır.")

        coverage = term_coverage("Güvenlik duvarı kuralları?", chunks)

        self.assertAlmostEqual(coverage, 1 / 3)

    def test_no_coverage(self):
        chunks = make_chunks("Kategorik ifadeler sayısal değerlere dönüşür.")

        self.assertEqual(
            term_coverage("Parola kaç karakter uzunluğunda?", chunks),
            0.0,
        )

    def test_weights_change_coverage_without_changing_matches(self):
        # Aynı eşleşmeler, farklı sonuç: eşit sayınca 2/3, ayırt ediciliğe göre
        # sayınca eksik kelime baskın çıkıyor.
        chunks = make_chunks("Güvenli yedekleme için 3-2-1 kuralı önerilir.")
        question = "Güvenlik duvarı kuralları nasıl olmalıdır?"
        weights = {"güvenlik": 0.97, "duvarı": 3.91, "kuralları": 2.30}

        self.assertAlmostEqual(term_coverage(question, chunks), 2 / 3)
        self.assertLess(term_coverage(question, chunks, weights=weights), 0.50)

    def test_missing_weight_defaults_to_one(self):
        chunks = make_chunks("Parola uzunluğu belirlenir.")

        coverage = term_coverage(
            "Parola sıklığı nedir?",
            chunks,
            weights={"parola": 1.0},
        )

        self.assertAlmostEqual(coverage, 0.5)

    def test_coverage_is_none_when_all_weights_are_zero(self):
        chunks = make_chunks("Parola uzunluğu belirlenir.")

        coverage = term_coverage(
            "Parola nedir?",
            chunks,
            weights={"parola": 0.0},
        )

        self.assertIsNone(coverage)

    def test_coverage_is_none_when_question_has_no_content_terms(self):
        self.assertIsNone(term_coverage("Bu ne?", make_chunks("herhangi bir metin")))


class EvidenceGateTests(unittest.TestCase):
    def test_evidence_present_passes(self):
        chunks = make_chunks(
            "Çok faktörlü kimlik doğrulama iki bağımsız kanıt ister."
        )

        self.assertTrue(
            has_term_evidence("Çok faktörlü kimlik doğrulama nedir?", chunks)
        )

    def test_evidence_absent_fails(self):
        chunks = make_chunks("Kategorik ifadeler sayısal değerlere dönüşür.")

        self.assertFalse(
            has_term_evidence("Parola kaç karakter uzunluğunda?", chunks)
        )

    def test_empty_chunks_fail(self):
        self.assertFalse(has_term_evidence("Herhangi bir soru?", []))

    def test_unmeasurable_question_is_allowed(self):
        # Ölçemediğimiz bir gerekçeyle kullanıcıyı reddetmek yanlış olur.
        self.assertTrue(has_term_evidence("Bu ne?", make_chunks("metin")))

    def test_threshold_is_respected(self):
        chunks = make_chunks("Güvenlik olayı müdahalesi anlatılır.")
        question = "Güvenlik duvarı kuralları?"

        self.assertFalse(has_term_evidence(question, chunks, threshold=0.50))
        self.assertTrue(has_term_evidence(question, chunks, threshold=0.30))


if __name__ == "__main__":
    unittest.main()
