import re
import time
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Callable

from app.config import (
    CONTEXT_RELATIVE_SCORE_MARGIN,
    CONTEXT_SCORE_THRESHOLD,
    CONTEXT_TERM_EVIDENCE_MIN,
    EXTRACTIVE_SCORE_THRESHOLD,
    EXTRACTIVE_TERM_EVIDENCE_MIN,
    GROUNDEDNESS_THRESHOLD,
    MAX_CONTEXT_CHUNKS,
    MAX_EXTRACTIVE_CHARS,
    NEIGHBOR_CHUNK_RADIUS,
    NO_EVIDENCE_ANSWER,
    SIMILARITY_THRESHOLD,
    TERM_EVIDENCE_MIN_PREFIX,
    TERM_EVIDENCE_THRESHOLD,
    TOP_K,
    USE_EXTRACTIVE_FALLBACK,
)
from app.groundedness import is_grounded
from app.llm import LocalLLM, get_answer_validation_error
from app.prompts import build_rag_messages
from app.retrieval import gate_score, get_top_chunks
from app.term_evidence import term_coverage


class EmptyQuestionError(ValueError):
    pass


class EmptyIndexError(RuntimeError):
    pass


QUESTION_STOP_WORDS = {
    "acaba",
    "bir",
    "bu",
    "hangi",
    "için",
    "ile",
    "mi",
    "mı",
    "mu",
    "mü",
    "nasıl",
    "ne",
    "nedir",
    "neden",
    "olarak",
    "şu",
    "ve",
    "veya",
}


def tokenize_for_matching(text):
    return [
        token
        for token in re.findall(r"[\w-]+", text.casefold(), flags=re.UNICODE)
        if token not in QUESTION_STOP_WORDS and len(token) >= 3
    ]


def tokens_match(left, right):
    if left == right:
        return True

    shorter, longer = sorted((left, right), key=len)
    return len(shorter) >= 4 and longer.startswith(shorter)


def sentence_question_overlap(question_terms, sentence):
    sentence_terms = tokenize_for_matching(sentence)
    return sum(
        any(tokens_match(question_term, sentence_term) for sentence_term in sentence_terms)
        for question_term in question_terms
    )


def build_extractive_fallback(question, chunks, max_chars=500):
    question_terms = tokenize_for_matching(question)
    candidates = []
    sentence_order = 0

    for chunk in chunks:
        sentences = re.split(r"(?<=[.!?])\s+", chunk["chunk_text"].strip())
        for sentence in sentences:
            clean_sentence = sentence.strip()
            if not clean_sentence:
                continue
            candidates.append({
                "text": clean_sentence,
                "overlap": sentence_question_overlap(question_terms, clean_sentence),
                "chunk_score": float(chunk["score"]),
                "order": sentence_order,
            })
            sentence_order += 1

    if not candidates:
        return chunks[0]["chunk_text"].strip()

    best_overlap = max(candidate["overlap"] for candidate in candidates)
    if best_overlap == 0:
        return chunks[0]["chunk_text"].strip()

    minimum_overlap = max(1, best_overlap - 1)
    selected = sorted(
        (
            candidate
            for candidate in candidates
            if candidate["overlap"] >= minimum_overlap
        ),
        key=lambda candidate: (
            -candidate["overlap"],
            -candidate["chunk_score"],
            candidate["order"],
        ),
    )[:3]
    selected.sort(key=lambda candidate: candidate["order"])

    answer_parts = []
    for candidate in selected:
        proposed = " ".join(answer_parts + [candidate["text"]])
        if len(proposed) > max_chars:
            continue
        answer_parts.append(candidate["text"])

    return " ".join(answer_parts) or selected[0]["text"][:max_chars].strip()


@dataclass(frozen=True)
class RAGSource:
    id: int
    source_name: str
    chunk_text: str
    score: float
    source_type: str | None = None
    page_number: int | None = None
    chunk_index: int | None = None
    context_role: str = "matched"

    @classmethod
    def from_chunk(cls, chunk):
        return cls(
            id=chunk["id"],
            source_name=chunk["source_name"],
            source_type=chunk.get("source_type"),
            page_number=chunk.get("page_number"),
            chunk_index=chunk.get("chunk_index"),
            chunk_text=chunk["chunk_text"],
            score=float(chunk["score"]),
            context_role=chunk.get("context_role", "matched"),
        )


@dataclass(frozen=True)
class RAGTimings:
    retrieval_seconds: float
    generation_seconds: float
    total_seconds: float


@dataclass(frozen=True)
class RAGResult:
    question: str
    answer: str
    mode: str
    best_score: float
    sources: tuple[RAGSource, ...]
    timings: RAGTimings
    source_filter: str | None = None
    warning: str | None = None
    warning_solution: str | None = None
    warning_error: Exception | None = field(default=None, repr=False, compare=False)
    prompt_messages: tuple[dict, ...] = field(default=(), repr=False)


class RAGService:
    def __init__(
        self,
        retrieval_func=get_top_chunks,
        llm_factory=LocalLLM,
        clock=time.perf_counter,
        top_k=TOP_K,
        similarity_threshold=SIMILARITY_THRESHOLD,
        context_score_threshold=CONTEXT_SCORE_THRESHOLD,
        context_relative_score_margin=CONTEXT_RELATIVE_SCORE_MARGIN,
        extractive_score_threshold=EXTRACTIVE_SCORE_THRESHOLD,
        max_extractive_chars=MAX_EXTRACTIVE_CHARS,
        use_extractive_fallback=USE_EXTRACTIVE_FALLBACK,
        neighbor_chunk_radius=NEIGHBOR_CHUNK_RADIUS,
        max_context_chunks=MAX_CONTEXT_CHUNKS,
        term_evidence_threshold=TERM_EVIDENCE_THRESHOLD,
        term_evidence_min_prefix=TERM_EVIDENCE_MIN_PREFIX,
        context_term_evidence_min=CONTEXT_TERM_EVIDENCE_MIN,
        groundedness_threshold=GROUNDEDNESS_THRESHOLD,
        extractive_term_evidence_min=EXTRACTIVE_TERM_EVIDENCE_MIN,
    ):
        self.retrieval_func = retrieval_func
        self.llm_factory = llm_factory
        self.clock = clock
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold
        self.context_score_threshold = context_score_threshold
        self.context_relative_score_margin = context_relative_score_margin
        self.extractive_score_threshold = extractive_score_threshold
        self.max_extractive_chars = max_extractive_chars
        self.use_extractive_fallback = use_extractive_fallback
        self.neighbor_chunk_radius = neighbor_chunk_radius
        self.max_context_chunks = max_context_chunks
        self.term_evidence_threshold = term_evidence_threshold
        self.term_evidence_min_prefix = term_evidence_min_prefix
        self.context_term_evidence_min = context_term_evidence_min
        self.groundedness_threshold = groundedness_threshold
        self.extractive_term_evidence_min = extractive_term_evidence_min

    def answer(
        self,
        question,
        source_name=None,
        activity_factory: Callable[[str], object] | None = None,
        context_callback: Callable[[str, tuple[RAGSource, ...], tuple[dict, ...]], None]
        | None = None,
        stream_callback: Callable[[str], None] | None = None,
    ):
        clean_question = question.strip()

        if not clean_question:
            raise EmptyQuestionError("Soru boş olamaz.")

        make_activity = activity_factory or (lambda _stage: nullcontext())
        total_start = self.clock()
        retrieval_start = self.clock()

        retrieval_kwargs = {
            "top_k": self.top_k,
            "neighbor_radius": self.neighbor_chunk_radius,
        }
        if source_name is not None:
            retrieval_kwargs["source_name"] = source_name

        with make_activity("retrieval"):
            chunks = self.retrieval_func(clean_question, **retrieval_kwargs)

        retrieval_seconds = self.clock() - retrieval_start

        if not chunks:
            raise EmptyIndexError("Aranabilecek bir indeks bulunamadı.")

        # Hybrid search sıralamayı değiştirdiği için chunks[0] en yüksek cosine
        # skoruna sahip chunk olmayabilir. Kapı skorunu sıralamadan okumak eşiği
        # sessizce kaydırır.
        best_score = gate_score(chunks)

        if best_score < self.similarity_threshold:
            return RAGResult(
                question=clean_question,
                answer=NO_EVIDENCE_ANSWER,
                mode="no_evidence",
                best_score=best_score,
                sources=(),
                timings=RAGTimings(
                    retrieval_seconds=retrieval_seconds,
                    generation_seconds=0.0,
                    total_seconds=self.clock() - total_start,
                ),
                source_filter=source_name,
            )

        matched_context_chunks = self.select_matched_context_chunks(
            chunks,
            question=clean_question,
        )
        matched_sources = tuple(
            RAGSource.from_chunk(chunk)
            for chunk in matched_context_chunks
        )

        term_weights = chunks[0].get("question_term_weights")
        use_extractive = self.should_use_extractive_answer(
            matched_sources,
        ) and self.has_extractive_term_evidence(
            clean_question,
            matched_context_chunks,
            term_weights,
        )

        if use_extractive:
            prompt_chunks = matched_context_chunks
            source_chunks = matched_context_chunks
        else:
            expanded_chunks = self.expand_context_chunks(matched_context_chunks)
            prompt_chunks = self.order_context_chunks(
                expanded_chunks,
                matched_context_chunks,
            )
            source_chunks = expanded_chunks

        if not self.has_term_evidence(clean_question, prompt_chunks, term_weights):
            return RAGResult(
                question=clean_question,
                answer=NO_EVIDENCE_ANSWER,
                mode="no_evidence",
                best_score=best_score,
                sources=(),
                timings=RAGTimings(
                    retrieval_seconds=retrieval_seconds,
                    generation_seconds=0.0,
                    total_seconds=self.clock() - total_start,
                ),
                source_filter=source_name,
            )

        sources = tuple(RAGSource.from_chunk(chunk) for chunk in source_chunks)
        messages = tuple(build_rag_messages(clean_question, prompt_chunks))

        if context_callback is not None:
            context_callback(clean_question, sources, messages)

        generation_start = self.clock()
        warning = None
        warning_solution = None
        warning_error = None

        if use_extractive:
            answer = matched_sources[0].chunk_text
            mode = "extractive"
        else:
            fallback_answer = build_extractive_fallback(
                clean_question,
                prompt_chunks,
                max_chars=self.max_extractive_chars,
            )
            answer, mode, warning, warning_error = self.generate_with_fallback(
                messages,
                fallback_answer,
                activity_factory=make_activity,
                stream_callback=stream_callback,
            )

            if warning_error is not None:
                warning_solution = "/doctor ile LLM durumunu kontrol et."

        generation_seconds = self.clock() - generation_start

        # Groundedness kontrolü yalnızca `generative` modda çalışır. `extractive`
        # ve `fallback_extractive` chunk metnini birebir döndürür, yani inşası
        # gereği dayanaklıdır; onları ölçmek her zaman 1.0 verir ve yalnızca
        # kalibrasyonu yanıltır.
        rejected_mode = None

        if mode == "no_evidence":
            rejected_mode = "no_evidence"
        elif mode == "generative" and not self.is_grounded(answer, prompt_chunks):
            rejected_mode = "ungrounded"
        elif mode == "fallback_extractive" and not self.has_extractive_term_evidence(
            clean_question,
            prompt_chunks,
            term_weights,
        ):
            # `fallback_extractive`, `extractive` ile AYNI iddiayı yapar: "bu
            # kaynak metni cevaptır". Bu yüzden aynı kanıt şartını taşımalıdır.
            # Groundedness burada işe yaramaz, çünkü metin zaten context'ten
            # geliyor ve tanım gereği 1.0 alır — dayanaklı olmak alakalı olmak
            # değildir. Manuel testte ölçülen sızıntı buydu: "Fidye yazılımının
            # şifrelediği dosyaları çözmek için hangi araç kullanılır?" sorusuna
            # model geçersiz bir üretim yaptı, akış fallback'e düştü ve alakasız
            # bir yedekleme cümlesi cevap olarak gösterildi (kapsama 0.32).
            rejected_mode = "no_evidence"

        if rejected_mode is not None:
            return RAGResult(
                question=clean_question,
                answer=NO_EVIDENCE_ANSWER,
                mode=rejected_mode,
                best_score=best_score,
                sources=(),
                timings=RAGTimings(
                    retrieval_seconds=retrieval_seconds,
                    generation_seconds=generation_seconds,
                    total_seconds=self.clock() - total_start,
                ),
                source_filter=source_name,
                prompt_messages=messages,
            )

        return RAGResult(
            question=clean_question,
            answer=answer,
            mode=mode,
            best_score=best_score,
            sources=sources,
            timings=RAGTimings(
                retrieval_seconds=retrieval_seconds,
                generation_seconds=generation_seconds,
                total_seconds=self.clock() - total_start,
            ),
            source_filter=source_name,
            warning=warning,
            warning_solution=warning_solution,
            warning_error=warning_error,
            prompt_messages=messages,
        )

    def has_extractive_term_evidence(self, question, chunks, weights=None):
        """Extractive kısayolu için gereken güçlü kelime kanıtı.

        Bu yol chunk metnini doğrudan cevap yapar; modelden de groundedness
        kontrolünden de geçmez. Ön kapı alan filtresine indirilince ölçümde iki
        hard negative tam buradan sızdı.

        Kanıt yetersizse soru reddedilmez, yalnızca üretken yola düşer; kararı
        orada model ve groundedness verir. Bu yüzden burada yüksek eşik
        savunulabilir: yanlış reddin bedeli bir mod değişikliğidir, cevapsızlık
        değil.
        """
        coverage = term_coverage(
            question,
            chunks,
            min_prefix=self.term_evidence_min_prefix,
            weights=weights,
        )

        if coverage is None:
            return True

        return coverage >= self.extractive_term_evidence_min

    def is_grounded(self, answer, chunks):
        """Üretilen cevap verilen context'e dayanıyor mu?

        Kelime kanıtı kapısı **sorunun** kelimelerine bakar ve 112 etiketli
        vakada ayırt ediciliğini kaybetti (meşru soru 0.27, tuzak 0.65). Sebep
        yapısaldır: kullanıcı soruyu kendi kelimeleriyle sorar. Bu kontrol aynı
        soruyu cevabın üstünden sorar; karşılaştırılan iki metin de kaynağın
        dilindedir, çünkü model cevabı context'ten okuyarak üretir.

        Bu kontrol generation'dan SONRA çalışır. Bedeli, kapsam dışı sorularda
        da modelin çalışmasıdır; karşılığı, kapının hiç göremediği uydurma
        cevabın yakalanmasıdır.
        """
        return is_grounded(answer, chunks, threshold=self.groundedness_threshold)

    def has_term_evidence(self, question, chunks, weights=None):
        """Sorunun kelimeleri seçilen context'te gerçekten geçiyor mu?

        Similarity skoru konu benzerliğini ölçer, cevabın metinde bulunup
        bulunmadığını değil. Konusu dokümana yakın ama cevabı dokümanda olmayan
        sorular bu yüzden similarity eşiğini rahatça geçebiliyor. Kelime kanıtı
        bağımsız bir sinyaldir ve bu iki durumu ayırır.

        Bu kontrol LLM çağrısından önce yapılır. Böylece hem kanıtsız soruya
        uydurma cevap üretilmesi engellenir, hem de `generate_with_fallback`
        içindeki yanlış ret koruması yalnızca gerçekten kanıt varken devreye
        girer; modelin doğru reddi artık ezilmez.

        `weights` retrieval'ın korpustan ölçtüğü ayırt edicilik ağırlıklarıdır.
        Verilmezse bütün kelimeler eşit sayılır; bu davranış ölçümde sızdırdı,
        bu yüzden uygulama yolunda ağırlıklar daima gelir.
        """
        coverage = term_coverage(
            question,
            chunks,
            min_prefix=self.term_evidence_min_prefix,
            weights=weights,
        )

        if coverage is None:
            return True

        return coverage >= self.term_evidence_threshold

    def should_use_extractive_answer(self, sources):
        if not self.use_extractive_fallback or len(sources) != 1:
            return False

        best_source = sources[0]
        return (
            best_source.score >= self.extractive_score_threshold
            and len(best_source.chunk_text) <= self.max_extractive_chars
        )

    def select_matched_context_chunks(self, chunks, question=None):
        # Göreli marj en yüksek cosine skoruna göre ölçülür, listenin ilk
        # elemanına göre değil: hybrid sıralamada ilk eleman daha düşük cosine
        # alabilir ve marj o kadar aşağı kayarak alakasız chunk'ları context'e
        # sokar.
        relative_threshold = (
            gate_score(chunks) - self.context_relative_score_margin
        )
        effective_threshold = max(
            self.context_score_threshold,
            relative_threshold,
        )
        # En iyi sıralanan chunk her zaman context'e girer. Eşik cosine üzerinden
        # çalışır, sıralama ise hybrid; ölçümde bu ikisi çelişti. "Saplama ile
        # taklit nesne arasındaki fark nedir?" sorusunda cevabı içeren chunk
        # 1. sıradaydı ama cosine'i 0.3147 olduğu için elendi, 3. sıradaki
        # alakasız chunk 0.3623 ile context'e girdi. Sıralamanın birincisini
        # elemek, hybrid search'ün kazandırdığı şeyi geri vermektir.
        matched_chunks = [chunks[0]] + [
            chunk
            for chunk in chunks[1:]
            if chunk["score"] >= effective_threshold
            and self.has_context_term_evidence(question, chunk, chunks[0])
        ]

        return [
            dict(chunk, context_role="matched")
            for chunk in matched_chunks
        ]

    def has_context_term_evidence(self, question, chunk, best_chunk):
        """Bu chunk, sorunun ayırt edici kelimelerinden bir şey taşıyor mu?

        Cosine eşiği konu benzerliğini ölçer ve alakasız bir dokümandan gelen
        parçayı da geçirebilir. Ölçümde context'e sızan yedi parçanın hiçbirinde
        kayda değer kelime kanıtı yoktu (en yükseği `0.267`), oysa cosine'leri
        `0.36`-`0.53` arasındaydı. "Çok faktörlü doğrulama neden önemli?"
        sorusunda `datamining.pdf` parçaları böyle girip modelin konuları
        karıştırmasına yol açtı.

        Kural yalnızca ikinci ve sonraki sıralara uygulanır. Birinci sıra
        koşulsuz girer, çünkü elimizdeki en iyi cevap adayı odur; onu kanıt
        şartıyla elemek retrieval kararını ikinci kez sorgulamak olur.

        `question` verilmezse filtre uygulanmaz. Böylece bu metodu soru
        bağlamı olmadan çağıran araçlar eski davranışı görür.
        """
        if question is None:
            return True

        weights = best_chunk.get("question_term_weights")
        coverage = term_coverage(
            question,
            [chunk],
            min_prefix=self.term_evidence_min_prefix,
            weights=weights,
        )

        if coverage is None:
            return True

        return coverage >= self.context_term_evidence_min

    def expand_context_chunks(self, matched_chunks):
        expanded_chunks = []
        seen_ids = set()

        for chunk in matched_chunks:
            if chunk["id"] in seen_ids:
                continue
            expanded_chunks.append(dict(chunk, context_role="matched"))
            seen_ids.add(chunk["id"])

        for chunk in matched_chunks:
            if len(expanded_chunks) >= self.max_context_chunks:
                break

            neighbors = chunk.get("neighbors", [])
            if self.neighbor_chunk_radius >= 0:
                neighbors = neighbors[: self.neighbor_chunk_radius * 2]

            for neighbor in neighbors:
                if len(expanded_chunks) >= self.max_context_chunks:
                    break
                if neighbor["id"] in seen_ids:
                    continue
                if neighbor["score"] < self.context_score_threshold:
                    continue
                expanded_chunks.append(dict(neighbor, context_role="neighbor"))
                seen_ids.add(neighbor["id"])

        return expanded_chunks[: self.max_context_chunks]

    @staticmethod
    def order_context_chunks(expanded_chunks, matched_chunks):
        source_priority = {}
        for chunk in matched_chunks:
            source_priority.setdefault(chunk["source_name"], len(source_priority))

        return sorted(
            expanded_chunks,
            key=lambda chunk: (
                source_priority.get(chunk["source_name"], len(source_priority)),
                chunk.get("page_number") or 0,
                chunk.get("chunk_index")
                if chunk.get("chunk_index") is not None
                else chunk["id"],
                chunk["id"],
            ),
        )

    def generate_with_fallback(
        self,
        messages,
        fallback_answer,
        activity_factory=None,
        stream_callback=None,
    ):
        fallback_answer = fallback_answer.strip()
        make_activity = activity_factory or (lambda _stage: nullcontext())

        try:
            with make_activity("model"):
                llm_client = self.llm_factory()
            with make_activity("generation"):
                if stream_callback is not None and hasattr(
                    llm_client,
                    "generate_answer_stream",
                ):
                    generated_answer = llm_client.generate_answer_stream(
                        list(messages),
                        on_update=stream_callback,
                    )
                else:
                    generated_answer = llm_client.generate_answer(list(messages))
        except Exception as error:
            return (
                fallback_answer,
                "fallback_extractive",
                "LLM yanıtı alınamadı; kaynak metin kullanıldı.",
                error,
            )

        # Model context'te cevap bulamadığını söylüyorsa bu nihai cevaptır.
        # Eskiden geçersiz sayılıp kaynak metne dönülüyordu; ayrıntı ve ölçülen
        # hata `app/llm.get_answer_validation_error()` içinde.
        if generated_answer.strip().casefold() == NO_EVIDENCE_ANSWER.casefold():
            return NO_EVIDENCE_ANSWER, "no_evidence", None, None

        validation_error = get_answer_validation_error(generated_answer)
        if validation_error is not None:
            return (
                fallback_answer,
                "fallback_extractive",
                "LLM cevabı yeterli bulunmadı; kaynak metin kullanıldı.",
                None,
            )

        return generated_answer, "generative", None, None
