import unittest
from contextlib import contextmanager
from unittest.mock import Mock

from app.rag_service import (
    EmptyIndexError,
    EmptyQuestionError,
    NO_EVIDENCE_ANSWER,
    RAGService,
    build_extractive_fallback,
)


def make_chunk(
    score=0.70,
    text="RAG, ilgili bilgiyi dokumanlardan bulur.",
    chunk_id=7,
    source_name="example.txt",
    page_number=None,
    chunk_index=1,
    neighbors=None,
):
    chunk = {
        "id": chunk_id,
        "source_name": source_name,
        "source_type": "pdf" if page_number is not None else "txt",
        "page_number": page_number,
        "chunk_index": chunk_index,
        "chunk_text": text,
        "score": score,
    }
    if neighbors is not None:
        chunk["neighbors"] = neighbors
    return chunk


class RAGServiceTests(unittest.TestCase):
    def test_extractive_fallback_selects_sentence_matching_question_terms(self):
        chunks = [
            make_chunk(
                score=0.60,
                text=(
                    "Şüpheli istekler resmi kanaldan doğrulanmalıdır. "
                    "Çok faktörlü doğrulama hesabı korur."
                ),
                chunk_id=2,
                chunk_index=2,
            ),
            make_chunk(
                score=0.48,
                text=(
                    "Mesajın gönderen adresi dikkatle kontrol edilmeli ve "
                    "bağlantının gerçek hedef adresi incelenmelidir."
                ),
                chunk_id=1,
                chunk_index=1,
            ),
        ]

        answer = build_extractive_fallback(
            "Gönderen adresi ve bağlantılar nasıl kontrol edilmelidir?",
            chunks,
        )

        self.assertIn("gönderen adresi", answer.casefold())
        self.assertIn("gerçek hedef adresi", answer.casefold())
        self.assertNotIn("çok faktörlü", answer.casefold())

    def test_empty_question_and_index_raise_domain_errors(self):
        service = RAGService(retrieval_func=lambda *_args, **_kwargs: [])

        with self.assertRaises(EmptyQuestionError):
            service.answer("   ")

        with self.assertRaises(EmptyIndexError):
            service.answer("RAG nedir?")

    def test_no_evidence_result_does_not_load_llm(self):
        llm_factory = Mock()
        service = RAGService(
            retrieval_func=lambda *_args, **_kwargs: [make_chunk(score=0.05)],
            llm_factory=llm_factory,
        )

        result = service.answer("Hava nasil?")

        self.assertEqual(result.answer, NO_EVIDENCE_ANSWER)
        self.assertEqual(result.mode, "no_evidence")
        self.assertEqual(result.sources, ())
        self.assertEqual(result.timings.generation_seconds, 0.0)
        llm_factory.assert_not_called()

    def test_high_score_without_term_evidence_is_rejected_before_llm(self):
        # Konusu dokümana yakın ama cevabı dokümanda olmayan soru. Similarity
        # eşiğini rahatça geçiyor; kelime kanıtı olmadığı için LLM'e hiç
        # gitmemeli, yoksa model eldeki alakasız metinden cevap uydurur.
        llm_factory = Mock()
        service = RAGService(
            retrieval_func=lambda *_args, **_kwargs: [
                make_chunk(
                    score=0.60,
                    text=(
                        "Kategorik ifadeler 0 ve 1 şeklinde sayısal "
                        "değerlere dönüştürülebilir."
                    ),
                ),
            ],
            llm_factory=llm_factory,
        )

        result = service.answer("Parola kaç karakter uzunluğunda olmalıdır?")

        self.assertEqual(result.mode, "no_evidence")
        self.assertEqual(result.answer, NO_EVIDENCE_ANSWER)
        self.assertEqual(result.sources, ())
        self.assertEqual(result.timings.generation_seconds, 0.0)
        llm_factory.assert_not_called()

    def test_term_evidence_gate_also_blocks_extractive_answers(self):
        # Kanıtsız bir soru yüksek skorlu tek bir chunk yakalarsa extractive
        # yoldan da geçmemeli; aksi halde alakasız metin doğrudan cevap olur.
        service = RAGService(
            retrieval_func=lambda *_args, **_kwargs: [
                make_chunk(
                    score=0.90,
                    text="Kategorik ifadeler sayısal değerlere dönüşür.",
                ),
            ],
            llm_factory=Mock(),
        )

        result = service.answer("Parola kaç karakter uzunluğunda olmalıdır?")

        self.assertEqual(result.mode, "no_evidence")

    def test_term_evidence_present_allows_normal_flow(self):
        service = RAGService(
            retrieval_func=lambda *_args, **_kwargs: [
                make_chunk(
                    score=0.90,
                    text=(
                        "Çok faktörlü kimlik doğrulama girişte iki bağımsız "
                        "kanıt ister."
                    ),
                ),
            ],
            llm_factory=Mock(),
        )

        result = service.answer("Çok faktörlü kimlik doğrulama nedir?")

        self.assertEqual(result.mode, "extractive")

    def test_term_evidence_threshold_is_configurable(self):
        chunks = [
            make_chunk(
                score=0.60,
                text="Güvenlik olayı müdahalesi anlatılır.",
            ),
        ]
        question = "Güvenlik duvarı kuralları nedir?"

        strict = RAGService(
            retrieval_func=lambda *_args, **_kwargs: chunks,
            llm_factory=Mock(),
            term_evidence_threshold=0.50,
        )
        lenient = RAGService(
            retrieval_func=lambda *_args, **_kwargs: chunks,
            llm_factory=Mock(),
            term_evidence_threshold=0.30,
        )

        self.assertEqual(strict.answer(question).mode, "no_evidence")
        self.assertNotEqual(lenient.answer(question).mode, "no_evidence")

    def test_term_weights_from_retrieval_close_a_measured_leak(self):
        # Ölçülen gerçek sızıntı: "güvenlik" neredeyse her chunk'ta geçtiği için
        # hiçbir şey kanıtlamaz, "kuralları" alakasız bir chunk'taki "3-2-1
        # kuralı" ile eşleşir, ayırt edici olan "duvarı" ise dokümanlarda hiç
        # yok. Eşit sayınca kapsama eşiği geçiyordu.
        chunks = [
            make_chunk(
                score=0.60,
                text="Güvenli yedekleme için 3-2-1 kuralı önerilir.",
            ),
        ]
        question = "Güvenlik duvarı kuralları nasıl olmalıdır?"
        weights = {"güvenlik": 0.97, "duvarı": 3.91, "kuralları": 2.30}

        # Eşik, ağırlıklandırmadan önceki kalibrasyona sabitlenir; böylece tek
        # değişken ağırlık olur ve test eşik değişince anlamını kaybetmez.
        service = RAGService(llm_factory=Mock(), term_evidence_threshold=0.60)

        self.assertTrue(service.has_term_evidence(question, chunks))
        self.assertFalse(service.has_term_evidence(question, chunks, weights))

    def test_retrieval_supplied_weights_reach_the_gate(self):
        # Ağırlıklar retrieval sonucundan okunur; bağlantı kopsa kapı sessizce
        # eşit sayma davranışına döner ve sızıntı geri gelir.
        chunk = make_chunk(
            score=0.60,
            text="Güvenli yedekleme için 3-2-1 kuralı önerilir.",
        )
        chunk["question_term_weights"] = {
            "güvenlik": 0.97,
            "duvarı": 3.91,
            "kuralları": 2.30,
        }
        service = RAGService(
            retrieval_func=lambda *_args, **_kwargs: [chunk],
            llm_factory=Mock(),
        )

        result = service.answer("Güvenlik duvarı kuralları nasıl olmalıdır?")

        self.assertEqual(result.mode, "no_evidence")

    def test_gate_score_ignores_hybrid_ordering(self):
        # Hybrid sıralamada ilk eleman daha düşük cosine alabilir. Kapı skorunu
        # listenin başından okumak eşiği sessizce kaydırır.
        chunks = [
            dict(
                make_chunk(score=0.18, chunk_id=1, chunk_index=1),
                dense_best_score=0.62,
            ),
            dict(
                make_chunk(score=0.62, chunk_id=2, chunk_index=2),
                dense_best_score=0.62,
            ),
        ]
        service = RAGService(
            retrieval_func=lambda *_args, **_kwargs: chunks,
            llm_factory=Mock(),
            similarity_threshold=0.20,
        )

        result = service.answer("RAG nedir?")

        self.assertNotEqual(result.mode, "no_evidence")
        self.assertAlmostEqual(result.best_score, 0.62)

    def test_extractive_result_is_structured_without_loading_llm(self):
        llm_factory = Mock()
        service = RAGService(
            retrieval_func=lambda *_args, **_kwargs: [make_chunk()],
            llm_factory=llm_factory,
        )

        result = service.answer("RAG nedir?")

        self.assertEqual(result.mode, "extractive")
        self.assertEqual(result.sources[0].id, 7)
        self.assertEqual(result.sources[0].source_name, "example.txt")
        self.assertGreaterEqual(result.timings.total_seconds, 0.0)
        llm_factory.assert_not_called()

    def test_extractive_answer_does_not_expand_neighbors(self):
        neighbor = make_chunk(score=0.22, chunk_id=8, chunk_index=2)
        matched = make_chunk(neighbors=[neighbor])
        service = RAGService(
            retrieval_func=lambda *_args, **_kwargs: [matched],
            llm_factory=Mock(),
        )

        result = service.answer("RAG nedir?")

        self.assertEqual(result.mode, "extractive")
        self.assertEqual([source.id for source in result.sources], [7])

    def test_source_filter_is_forwarded_to_retrieval(self):
        retrieval = Mock(return_value=[make_chunk(score=0.05)])
        service = RAGService(retrieval_func=retrieval)

        result = service.answer("RAG nedir?", source_name="example.txt")

        retrieval.assert_called_once_with(
            "RAG nedir?",
            top_k=3,
            neighbor_radius=1,
            source_name="example.txt",
        )
        self.assertEqual(result.source_filter, "example.txt")

    def test_generation_falls_back_and_keeps_warning_details(self):
        class BrokenLLM:
            def generate_answer(self, _messages):
                raise RuntimeError("model baglantisi koptu")

        chunks = [
            make_chunk(score=0.62, text="A" * 510, chunk_id=1, chunk_index=1),
            make_chunk(score=0.55, chunk_id=2, chunk_index=2),
        ]
        service = RAGService(
            retrieval_func=lambda *_args, **_kwargs: chunks,
            llm_factory=BrokenLLM,
        )

        result = service.answer("RAG nasıl çalışır?")

        self.assertEqual(result.mode, "fallback_extractive")
        # Fallback, soru terimleriyle en çok örtüşen cümleyi seçer; dolgu
        # metnini değil soruyla ilgili cümleyi döndürmesi beklenir.
        self.assertEqual(result.answer, "RAG, ilgili bilgiyi dokumanlardan bulur.")
        self.assertIn("kaynak metin", result.warning)
        self.assertIsInstance(result.warning_error, RuntimeError)

    def test_model_no_evidence_answer_falls_back_when_retrieval_has_evidence(self):
        class RejectingLLM:
            def generate_answer(self, _messages):
                return NO_EVIDENCE_ANSWER

        evidence = "3-2-1 kuralı verinin üç kopyasını iki ortamda tutmayı önerir."
        chunks = [
            make_chunk(score=0.71, text=evidence),
            make_chunk(score=0.52, chunk_id=8),
        ]
        service = RAGService(
            retrieval_func=lambda *_args, **_kwargs: chunks,
            llm_factory=RejectingLLM,
        )

        result = service.answer("3-2-1 kuralı nedir?")

        self.assertEqual(result.mode, "fallback_extractive")
        self.assertEqual(result.answer, evidence)
        self.assertIn("bulunan kanıtı kullanmadı", result.warning)

    def test_activity_and_context_hooks_cover_all_stages(self):
        stages = []
        context_calls = []

        @contextmanager
        def record_activity(stage):
            stages.append(stage)
            yield

        class GoodLLM:
            def generate_answer(self, _messages):
                return "Dokumanlara dayali yeterince uzun ve gecerli cevap."

        chunks = [
            make_chunk(score=0.62, text="A" * 510, chunk_id=1, chunk_index=1),
            make_chunk(score=0.55, chunk_id=2, chunk_index=2),
        ]
        service = RAGService(
            retrieval_func=lambda *_args, **_kwargs: chunks,
            llm_factory=GoodLLM,
        )

        result = service.answer(
            "RAG nasıl çalışır?",
            activity_factory=record_activity,
            context_callback=lambda *args: context_calls.append(args),
        )

        self.assertEqual(result.mode, "generative")
        self.assertEqual(stages, ["retrieval", "model", "generation"])
        self.assertEqual(len(context_calls), 1)
        self.assertEqual(context_calls[0][0], "RAG nasıl çalışır?")

    def test_prompt_context_is_ordered_by_document_after_relevance_selection(self):
        captured_messages = []

        class CapturingLLM:
            def generate_answer(self, messages):
                captured_messages.extend(messages)
                return "Doküman sırasına dayalı yeterince uzun ve geçerli cevap."

        chunks = [
            make_chunk(
                score=0.72,
                text="İkinci sayfadaki sonuç bölümü süreci özetler.",
                chunk_id=2,
                page_number=2,
                chunk_index=1,
            ),
            make_chunk(
                score=0.61,
                text="Birinci sayfadaki başlangıç bölümü süreci tanıtır.",
                chunk_id=1,
                page_number=1,
                chunk_index=1,
            ),
        ]
        service = RAGService(
            retrieval_func=lambda *_args, **_kwargs: chunks,
            llm_factory=CapturingLLM,
        )

        result = service.answer("Süreç nedir?")
        prompt = captured_messages[1]["content"]

        self.assertLess(prompt.index("başlangıç"), prompt.index("sonuç"))
        self.assertEqual([source.id for source in result.sources], [2, 1])

    def test_generative_context_adds_limited_scored_neighbors(self):
        previous_chunk = make_chunk(
            score=0.40,
            text="Önceki açıklama.",
            chunk_id=10,
            page_number=1,
            chunk_index=1,
        )
        next_chunk = make_chunk(
            score=0.38,
            text="Sonraki açıklama.",
            chunk_id=12,
            page_number=1,
            chunk_index=3,
        )
        matched = make_chunk(
            score=0.70,
            text="Ana süreç açıklaması. " + "A" * 510,
            chunk_id=11,
            page_number=1,
            chunk_index=2,
            neighbors=[previous_chunk, next_chunk],
        )

        class GoodLLM:
            def generate_answer(self, _messages):
                return "Komşu bağlama dayalı yeterince uzun ve geçerli cevap."

        service = RAGService(
            retrieval_func=lambda *_args, **_kwargs: [matched],
            llm_factory=GoodLLM,
            max_context_chunks=3,
        )

        result = service.answer("Ana süreç nedir?")

        self.assertEqual([source.id for source in result.sources], [11, 10, 12])
        self.assertEqual(
            [source.context_role for source in result.sources],
            ["matched", "neighbor", "neighbor"],
        )

    def test_context_uses_absolute_and_best_score_relative_thresholds(self):
        chunks = [
            make_chunk(score=0.71, chunk_id=1),
            make_chunk(score=0.52, chunk_id=2),
            make_chunk(score=0.49, chunk_id=3),
            make_chunk(score=0.34, chunk_id=4),
        ]
        service = RAGService(
            context_score_threshold=0.35,
            context_relative_score_margin=0.20,
        )

        selected = service.select_matched_context_chunks(chunks)

        self.assertEqual([chunk["id"] for chunk in selected], [1, 2])

    def test_neighbor_below_context_threshold_is_not_added(self):
        weak_neighbor = make_chunk(score=0.19, chunk_id=9, chunk_index=2)
        matched = make_chunk(
            score=0.70,
            chunk_id=8,
            chunk_index=1,
            neighbors=[weak_neighbor],
        )
        service = RAGService(context_score_threshold=0.35)

        expanded = service.expand_context_chunks([matched])

        self.assertEqual([chunk["id"] for chunk in expanded], [8])

    def test_streaming_llm_forwards_updates_and_returns_final_answer(self):
        updates = []

        class StreamingLLM:
            def generate_answer_stream(self, _messages, on_update):
                on_update("İlk parça")
                on_update("Tam ve dokümana dayalı geçerli cevap metni.")
                return "Tam ve dokümana dayalı geçerli cevap metni."

        chunks = [
            make_chunk(score=0.62, text="A" * 510, chunk_id=1, chunk_index=1),
            make_chunk(score=0.55, chunk_id=2, chunk_index=2),
        ]
        service = RAGService(
            retrieval_func=lambda *_args, **_kwargs: chunks,
            llm_factory=StreamingLLM,
        )

        result = service.answer(
            "RAG nasıl çalışır?",
            stream_callback=updates.append,
        )

        self.assertEqual(result.mode, "generative")
        self.assertEqual(updates, [
            "İlk parça",
            "Tam ve dokümana dayalı geçerli cevap metni.",
        ])

    def test_keyboard_interrupt_is_not_converted_to_fallback(self):
        class CancelledLLM:
            def generate_answer_stream(self, _messages, on_update):
                on_update("Yarım cevap")
                raise KeyboardInterrupt

        chunks = [
            make_chunk(score=0.62, text="A" * 510, chunk_id=1, chunk_index=1),
            make_chunk(score=0.55, chunk_id=2, chunk_index=2),
        ]
        service = RAGService(
            retrieval_func=lambda *_args, **_kwargs: chunks,
            llm_factory=CancelledLLM,
        )

        with self.assertRaises(KeyboardInterrupt):
            service.answer(
                "RAG nasıl çalışır?",
                stream_callback=lambda _answer: None,
            )


if __name__ == "__main__":
    unittest.main()
