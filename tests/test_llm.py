import subprocess
import unittest
from unittest.mock import MagicMock, patch

from app.llm import clean_answer, create_foundry_manager, is_valid_answer


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


class FoundryStartupTests(unittest.TestCase):
    def test_running_service_does_not_start_a_new_process(self):
        manager = MagicMock()
        manager.is_service_running.return_value = True

        with patch("app.llm.FoundryLocalManager", return_value=manager) as manager_class:
            with patch("app.llm.subprocess.Popen") as popen:
                result = create_foundry_manager()

        self.assertIs(result, manager)
        manager_class.assert_called_once_with(bootstrap=False)
        popen.assert_not_called()

    def test_service_start_process_is_silent_in_normal_mode(self):
        manager = MagicMock()
        manager.is_service_running.side_effect = [False, True]

        with patch("app.llm.FoundryLocalManager", return_value=manager):
            with patch("app.llm.subprocess.Popen") as popen:
                with patch("app.llm.time.sleep"):
                    result = create_foundry_manager()

        self.assertIs(result, manager)
        popen.assert_called_once_with(
            ["foundry", "service", "start"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    def test_debug_mode_uses_foundry_default_startup_output(self):
        manager = MagicMock()

        with patch("app.llm.FoundryLocalManager", return_value=manager) as manager_class:
            with patch("app.llm.subprocess.Popen") as popen:
                result = create_foundry_manager(show_startup_output=True)

        self.assertIs(result, manager)
        manager_class.assert_called_once_with()
        popen.assert_not_called()

    def test_service_start_timeout_raises_clear_error(self):
        manager = MagicMock()
        manager.is_service_running.return_value = False

        with patch("app.llm.FOUNDRY_START_ATTEMPTS", 2):
            with patch("app.llm.FoundryLocalManager", return_value=manager):
                with patch("app.llm.subprocess.Popen"):
                    with patch("app.llm.time.sleep"):
                        with self.assertRaisesRegex(
                            RuntimeError,
                            "Foundry Local servisi zamanında başlatılamadı",
                        ):
                            create_foundry_manager()


if __name__ == "__main__":
    unittest.main()
