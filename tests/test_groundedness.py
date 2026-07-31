import unittest
from unittest.mock import Mock

from app.config import NO_EVIDENCE_ANSWER
from app.groundedness import (
    groundedness_score,
    is_grounded,
    sentence_support,
    split_sentences,
)
from app.rag_service import RAGService
from app.term_evidence import build_context_terms


def make_chunk(text, score=0.70, chunk_id=1, chunk_index=1):
    return {
        "id": chunk_id,
        "source_name": "example.txt",
        "source_type": "txt",
        "page_number": None,
        "chunk_index": chunk_index,
        "chunk_text": text,
        "score": score,
    }


CONTEXT = [
    make_chunk(
        "Kilitlenme, iki sürecin birbirinin tuttuğu kaynağı beklemesiyle "
        "oluşur. Kaynakları daima aynı sırada istemek bu durumu önler.",
    ),
]


class SplitSentencesTests(unittest.TestCase):
    def test_splits_on_turkish_sentence_punctuation(self):
        self.assertEqual(
            split_sentences("Birinci cümle. İkinci cümle! Üçüncü cümle?"),
            ["Birinci cümle.", "İkinci cümle!", "Üçüncü cümle?"],
        )

    def test_empty_text_produces_no_sentences(self):
        self.assertEqual(split_sentences("   "), [])


class SentenceSupportTests(unittest.TestCase):
    def test_sentence_copied_from_context_is_fully_supported(self):
        terms = build_context_terms(CONTEXT)

        self.assertEqual(
            sentence_support("Kaynakları daima aynı sırada istemek gerekir.", terms),
            1.0,
        )

    def test_unrelated_sentence_has_no_support(self):
        terms = build_context_terms(CONTEXT)

        self.assertEqual(
            sentence_support("Çikolatalı kek fırında pişirilir.", terms),
            0.0,
        )

    def test_sentence_without_content_words_is_not_measurable(self):
        """Ölçemediğimiz bir gerekçeyle kullanıcıyı reddetmek yanlış olur."""
        terms = build_context_terms(CONTEXT)

        self.assertIsNone(sentence_support("Ve bu da şu.", terms))

    def test_turkish_inflection_does_not_break_support(self):
        """Cevap kaynağın kelimelerini çekimleyerek kullanır.

        `terms_match()` ortak kök temellidir; bu bağ kopsa dayanaklı cevaplar
        Türkçe ekleri yüzünden reddedilirdi. Buradaki cevap context'teki
        `kilitlenme`, `kaynağı` ve `süreçlerin` kelimelerini çekimli kullanır.
        """
        terms = build_context_terms(CONTEXT)

        self.assertEqual(
            sentence_support("Kilitlenmede süreçler kaynakları bekler.", terms),
            1.0,
        )

    def test_known_matcher_limit_survives_on_the_answer_side(self):
        """`önlemek` ~ `önler` ortak öneki 4 karakter, `min_prefix` 5.

        Bu, kelime kanıtı kapısını çökerten sınıfın ta kendisidir ve cevap
        tarafında da geçerlidir. Fark ölçekte: kapı SORUYU ölçüyordu ve soru
        yalnızca birkaç kelimeden oluştuğu için tek bir kaçırma kapsamayı
        eşiğin altına atıyordu. Cevap kaynaktan kopyalayarak yazıldığı için
        aynı kaçırma cümlenin geri kalanı tarafından taşınır; ölçümde parafraz
        cümlelerinin %96.2'si 0.60 eşiğini geçti.
        """
        terms = build_context_terms(CONTEXT)

        self.assertEqual(
            sentence_support("Kilitlenmeyi önlemek kaynak sırasıyla olur.", terms),
            0.5,
        )


class GroundednessScoreTests(unittest.TestCase):
    def test_answer_drawn_from_context_scores_one(self):
        answer = (
            "Kilitlenme iki sürecin birbirini beklemesiyle oluşur. "
            "Kaynakları aynı sırada istemek bunu önler."
        )

        self.assertEqual(groundedness_score(answer, CONTEXT), 1.0)

    def test_fabricated_answer_scores_zero(self):
        answer = (
            "Çikolatalı kek fırında pişirilir. "
            "Hamura kabartma tozu eklenmelidir."
        )

        self.assertEqual(groundedness_score(answer, CONTEXT), 0.0)

    def test_single_fabricated_sentence_lowers_a_grounded_answer(self):
        """Uydurma pratikte doğru cümlelerin arasına sıkışmış tek cümledir.

        Cevabın tamamını tek blok saymak bu cümleyi geçirirdi; ölçüm bu yüzden
        cümle bazlıdır.
        """
        answer = (
            "Kilitlenme iki sürecin birbirini beklemesiyle oluşur. "
            "Kaynakları aynı sırada istemek bunu önler. "
            "Çikolatalı kek fırında pişirilir."
        )

        self.assertAlmostEqual(groundedness_score(answer, CONTEXT), 2 / 3)

    def test_missing_context_is_not_measurable(self):
        self.assertIsNone(groundedness_score("Herhangi bir cevap.", []))

    def test_unmeasurable_answer_is_treated_as_grounded(self):
        self.assertTrue(is_grounded("Ve bu da şu.", CONTEXT))


class GroundednessInServiceTests(unittest.TestCase):
    """Kapının akıştaki yeri ve hangi modlara uygulandığı."""

    def test_ungrounded_generative_answer_is_rejected(self):
        class HallucinatingLLM:
            def generate_answer(self, _messages):
                return (
                    "Çikolatalı kek fırında pişirilir. "
                    "Hamura kabartma tozu eklenmelidir."
                )

        service = RAGService(
            retrieval_func=lambda *_args, **_kwargs: [
                make_chunk("A" * 510),
                make_chunk(
                    "Kilitlenme iki sürecin birbirini beklemesidir.",
                    score=0.55,
                    chunk_id=2,
                    chunk_index=2,
                ),
            ],
            llm_factory=HallucinatingLLM,
        )

        result = service.answer("Kilitlenme nedir ve nasıl önlenir?")

        self.assertEqual(result.mode, "ungrounded")
        self.assertEqual(result.answer, NO_EVIDENCE_ANSWER)
        self.assertEqual(result.sources, ())

    def test_grounded_generative_answer_passes(self):
        class GoodLLM:
            def generate_answer(self, _messages):
                return "Kilitlenme iki sürecin birbirini beklemesidir."

        service = RAGService(
            retrieval_func=lambda *_args, **_kwargs: [
                make_chunk("A" * 510),
                make_chunk(
                    "Kilitlenme iki sürecin birbirini beklemesidir.",
                    score=0.55,
                    chunk_id=2,
                    chunk_index=2,
                ),
            ],
            llm_factory=GoodLLM,
        )

        result = service.answer("Kilitlenme nedir ve nasıl önlenir?")

        self.assertEqual(result.mode, "generative")

    def test_extractive_answer_skips_the_groundedness_check(self):
        """`extractive` chunk metnini birebir döndürür; inşası gereği dayanaklı.

        Kontrolü oraya da uygulamak her zaman 1.0 verir ve yalnızca kalibrasyonu
        yanıltır.
        """
        service = RAGService(
            retrieval_func=lambda *_args, **_kwargs: [
                make_chunk(
                    "Kilitlenme iki sürecin birbirini beklemesidir.",
                    score=0.90,
                ),
            ],
            llm_factory=Mock(),
            groundedness_threshold=1.0,
        )

        result = service.answer("Kilitlenme nedir?")

        self.assertEqual(result.mode, "extractive")

    def test_weak_evidence_downgrades_extractive_instead_of_answering(self):
        """Ölçülen sızıntı: extractive kısayolu her iki kapıyı da atlıyordu.

        Bu yol chunk metnini doğrudan cevap yapar; ne modele ne groundedness'a
        uğrar. Ön kapı 0.675'ten 0.21'e indirilince iki hard negative tam
        buradan sızdı ve alakasız chunk metni cevap oldu.

        Doğru davranış reddetmek değil, üretken yola düşmektir; kararı orada
        model verir.
        """
        class RefusingLLM:
            def generate_answer(self, _messages):
                return NO_EVIDENCE_ANSWER

        service = RAGService(
            retrieval_func=lambda *_args, **_kwargs: [
                make_chunk(
                    "Sürüm kontrolünde değişiklikler dallar üzerinde tutulur.",
                    score=0.90,
                ),
            ],
            llm_factory=RefusingLLM,
        )

        result = service.answer("Git geri almak için hangi komut yazılır?")

        self.assertEqual(result.mode, "no_evidence")

    def test_fallback_extractive_needs_the_same_evidence_as_extractive(self):
        """Manuel testte ölçülen sızıntı: fallback yolu kanıtsız çalışıyordu.

        Model geçersiz bir üretim yaptığında akış kaynak metnine döner. O metin
        context'ten geldiği için groundedness onu her zaman dayanaklı bulur —
        ama dayanaklı olmak ALAKALI olmak değildir. "Fidye yazılımının
        şifrelediği dosyaları çözmek için hangi araç kullanılır?" sorusuna
        sistem bu yoldan alakasız bir yedekleme cümlesi gösterdi.

        `fallback_extractive` ile `extractive` aynı iddiayı yapar ("bu kaynak
        metni cevaptır"), bu yüzden aynı kanıt şartını taşır.
        """
        class FailingLLM:
            def generate_answer(self, _messages):
                return "Kısa."

        service = RAGService(
            retrieval_func=lambda *_args, **_kwargs: [
                make_chunk(
                    "Yedekleme, fidye yazılımı veya donanım arızası sonrasında "
                    "veriyi geri getirebilmek için yapılır.",
                    score=0.62,
                ),
                make_chunk(
                    "Kopyalardan biri farklı bir konumda saklanmalıdır.",
                    score=0.55,
                    chunk_id=2,
                    chunk_index=2,
                ),
            ],
            llm_factory=FailingLLM,
        )

        result = service.answer(
            "Fidye yazılımının şifrelediği dosyaları çözmek için hangi araç "
            "kullanılır?"
        )

        self.assertEqual(result.mode, "no_evidence")
        self.assertEqual(result.answer, NO_EVIDENCE_ANSWER)
        self.assertEqual(result.sources, ())

    def test_fallback_extractive_survives_when_evidence_is_strong(self):
        """Kanıt güçlüyse fallback hâlâ çalışmalı; aksi halde model bir kez
        bozuk üretim yaptığında meşru sorular da cevapsız kalırdı."""
        class FailingLLM:
            def generate_answer(self, _messages):
                return "Kısa."

        service = RAGService(
            retrieval_func=lambda *_args, **_kwargs: [
                make_chunk(
                    "Kilitlenme, iki sürecin birbirinin kaynağını beklemesidir.",
                    score=0.62,
                ),
                make_chunk(
                    "Kaynakları aynı sırada istemek kilitlenmeyi önler.",
                    score=0.55,
                    chunk_id=2,
                    chunk_index=2,
                ),
            ],
            llm_factory=FailingLLM,
        )

        result = service.answer("Kilitlenme nedir?")

        self.assertEqual(result.mode, "fallback_extractive")

    def test_strong_evidence_still_uses_the_extractive_shortcut(self):
        service = RAGService(
            retrieval_func=lambda *_args, **_kwargs: [
                make_chunk(
                    "Kilitlenme iki sürecin birbirini beklemesidir.",
                    score=0.90,
                ),
            ],
            llm_factory=Mock(),
        )

        result = service.answer("Kilitlenme nedir?")

        self.assertEqual(result.mode, "extractive")

    def test_model_refusal_is_final_and_not_replaced_by_source_text(self):
        """Ölçülen hata: modelin doğru reddi silinip alakasız metin gösteriliyordu.

        Ön kapı alan filtresine indirildikten sonra kapsam dışı sorular modele
        ulaşıyor; orada arama yanlış, model haklıdır.
        """
        class RefusingLLM:
            def generate_answer(self, _messages):
                return NO_EVIDENCE_ANSWER

        service = RAGService(
            retrieval_func=lambda *_args, **_kwargs: [
                make_chunk("A" * 510),
                make_chunk(
                    "Kilitlenme iki sürecin birbirini beklemesidir.",
                    score=0.55,
                    chunk_id=2,
                    chunk_index=2,
                ),
            ],
            llm_factory=RefusingLLM,
        )

        result = service.answer("Kilitlenme nedir ve nasıl önlenir?")

        self.assertEqual(result.mode, "no_evidence")
        self.assertEqual(result.answer, NO_EVIDENCE_ANSWER)
        self.assertEqual(result.sources, ())
        self.assertIsNone(result.warning)


if __name__ == "__main__":
    unittest.main()
