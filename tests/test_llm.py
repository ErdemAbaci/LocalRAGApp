import subprocess
import unittest
from unittest.mock import MagicMock, patch

from app.llm import (
    DEFAULT_MODEL_ALIAS,
    LocalLLM,
    clean_answer,
    create_foundry_manager,
    get_model_alias,
    get_model_alias_source,
    get_foundry_service_uri,
    has_excessive_repetition,
    is_valid_answer,
)


class ModelConfigurationTests(unittest.TestCase):
    def test_default_and_environment_model_aliases_are_resolved(self):
        self.assertEqual(get_model_alias({}), DEFAULT_MODEL_ALIAS)
        self.assertEqual(get_model_alias({"LOCAL_RAG_MODEL": " phi-3.5-mini "}), "phi-3.5-mini")
        self.assertEqual(get_model_alias({"LOCAL_RAG_MODEL": "  "}), DEFAULT_MODEL_ALIAS)
        self.assertEqual(get_model_alias_source({}), "varsayılan")
        self.assertEqual(
            get_model_alias_source({"LOCAL_RAG_MODEL": "phi-3.5-mini"}),
            "LOCAL_RAG_MODEL",
        )

    def test_local_llm_loads_explicit_model_alias(self):
        manager = MagicMock()
        manager.load_model.return_value.id = "loaded-model-id"

        with patch("app.llm.create_foundry_manager", return_value=manager):
            with patch("app.llm.openai.OpenAI") as openai_client:
                llm = LocalLLM(model_alias="phi-3.5-mini")

        self.assertEqual(llm.model_alias, "phi-3.5-mini")
        manager.load_model.assert_called_once_with("phi-3.5-mini")
        openai_client.assert_called_once_with(
            base_url=manager.endpoint,
            api_key=manager.api_key,
            timeout=120,
        )


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

    def test_repeated_word_loop_is_rejected(self):
        answer = "Gönderinin " * 18

        self.assertTrue(has_excessive_repetition(answer))
        self.assertFalse(is_valid_answer(answer))

    def test_normal_repeated_technical_terms_are_not_rejected(self):
        answer = (
            "Veri temizleme veriyi düzenler. Veri bütünleştirme farklı kaynakları "
            "birleştirir ve analiz için güvenilir bir veri kümesi hazırlar."
        )

        self.assertFalse(has_excessive_repetition(answer))
        self.assertTrue(is_valid_answer(answer))


class FoundryStartupTests(unittest.TestCase):
    def test_running_service_does_not_start_a_new_process(self):
        manager = MagicMock()
        manager.is_service_running.return_value = True

        with patch("app.llm.FoundryLocalManager", return_value=manager) as manager_class:
            with patch("app.llm.subprocess.Popen") as popen:
                result = create_foundry_manager()

        self.assertIs(result, manager)
        manager_class.assert_called_once_with(bootstrap=False, timeout=120)
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
        manager.is_service_running.side_effect = [False, True]

        with patch("app.llm.FoundryLocalManager", return_value=manager) as manager_class:
            with patch("app.llm.subprocess.Popen") as popen:
                with patch("app.llm.time.sleep"):
                    result = create_foundry_manager(show_startup_output=True)

        self.assertIs(result, manager)
        manager_class.assert_called_once_with(bootstrap=False, timeout=120)
        popen.assert_called_once_with(["foundry", "service", "start"])

    def test_service_status_timeout_raises_clear_error(self):
        with patch(
            "app.llm.subprocess.run",
            side_effect=subprocess.TimeoutExpired("foundry", 15),
        ):
            with self.assertRaisesRegex(RuntimeError, "servis durumu zamanında"):
                get_foundry_service_uri()

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
