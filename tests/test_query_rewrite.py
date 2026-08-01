import unittest

from app.query_rewrite import (
    FOLLOW_UP_MARKERS,
    FollowUpContext,
    carried_terms,
    content_terms,
    is_follow_up,
)
from app.term_evidence import normalize_text


class ContentTermsTest(unittest.TestCase):
    def test_marker_words_are_not_content(self):
        self.assertEqual(content_terms("Bunun maliyeti nedir?"), ["maliyeti"])

    def test_topic_word_survives(self):
        self.assertEqual(content_terms("Kilitlenme nedir?"), ["kilitlenme"])


class IsFollowUpTest(unittest.TestCase):
    def test_question_without_topic_word_is_follow_up(self):
        self.assertTrue(is_follow_up("Nasıl önlenir?"))

    def test_marker_with_single_topic_word_is_follow_up(self):
        self.assertTrue(is_follow_up("Peki maliyeti nedir?"))

    def test_pronoun_question_is_follow_up(self):
        self.assertTrue(is_follow_up("Bunun maliyeti nedir?"))

    def test_single_topic_word_without_marker_is_not_follow_up(self):
        # Tek kelimelik ama kendi başına yeterli soru. İşaret kelimesi şartı
        # olmasaydı "RAG nedir?" de takip sayılır ve önceki konunun kelimeleri
        # aramaya sızardı.
        self.assertFalse(is_follow_up("Kilitlenme nedir?"))

    def test_marker_with_enough_topic_words_is_not_follow_up(self):
        self.assertFalse(is_follow_up("Peki fidye yazılımı nedir?"))


class CarriedTermsTest(unittest.TestCase):
    def test_terms_come_from_previous_question(self):
        self.assertEqual(
            carried_terms("Nasıl önlenir?", "Kilitlenme nedir?"),
            ("kilitlenme",),
        )

    def test_already_present_root_is_not_carried(self):
        # `kilitlenme` ile `kilitlenmeyi` aynı kelimedir; birebir karşılaştırma
        # bunu göremez ve kelimeyi ikinci kez eklerdi.
        self.assertEqual(
            carried_terms("Kilitlenmeyi nasıl önlerim?", "Kilitlenme nedir?"),
            (),
        )

    def test_carry_count_is_capped(self):
        carried = carried_terms(
            "Nasıl önlenir?",
            "Dağıtık sistemlerde kilitlenme çakışma sorunu nedir?",
            max_terms=2,
        )
        self.assertEqual(len(carried), 2)


class FollowUpContextTest(unittest.TestCase):
    def test_first_question_is_untouched(self):
        context = FollowUpContext()
        self.assertEqual(
            context.resolve("Nasıl önlenir?"),
            ("Nasıl önlenir?", ()),
        )

    def test_follow_up_gets_previous_topic(self):
        context = FollowUpContext()
        context.remember("Kilitlenme nedir?")
        rewritten, carried = context.resolve("Peki nasıl önlenir?")
        self.assertEqual(carried, ("kilitlenme",))
        self.assertTrue(rewritten.startswith("Peki nasıl önlenir?"))
        self.assertIn("kilitlenme", rewritten)

    def test_self_sufficient_question_is_untouched(self):
        context = FollowUpContext()
        context.remember("Kilitlenme nedir?")
        self.assertEqual(
            context.resolve("Aşırı öğrenme nasıl anlaşılır?"),
            ("Aşırı öğrenme nasıl anlaşılır?", ()),
        )

    def test_topic_survives_a_chain_of_follow_ups(self):
        # Zincirin ikinci adımında konu ham sorudan okunsaydı `kilitlenme`
        # kaybolurdu; bu yüzden hatırlanan şey yeniden yazılmış sorudur.
        context = FollowUpContext()
        context.remember("Kilitlenme nedir?")
        rewritten, _ = context.resolve("Peki nasıl önlenir?")
        context.remember(rewritten)
        _, carried = context.resolve("Ya maliyeti?")
        self.assertIn("kilitlenme", carried)

    def test_clear_forgets_the_topic(self):
        context = FollowUpContext()
        context.remember("Kilitlenme nedir?")
        context.clear()
        self.assertEqual(context.resolve("Peki nasıl önlenir?")[1], ())

    def test_empty_question_is_not_remembered(self):
        context = FollowUpContext()
        context.remember("Kilitlenme nedir?")
        context.remember("   ")
        self.assertEqual(context.topic, "Kilitlenme nedir?")


class MarkerListTest(unittest.TestCase):
    def test_markers_are_written_in_turkish(self):
        # Liste normalize_text() çıktısıyla karşılaştırılır; ASCII yazılmış bir
        # kelime hiç eşleşmez ve markeri sessizce etkisiz bırakır.
        for marker in FOLLOW_UP_MARKERS:
            self.assertEqual(marker, normalize_text(marker))


if __name__ == "__main__":
    unittest.main()
