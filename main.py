import time

from app.retrieval import get_top_chunks
from app.prompts import build_rag_messages
from app.llm import LocalLLM
from app.ingest import ingest_documents

SIMILARITY_THRESHOLD = 0.20
CONTEXT_SCORE_THRESHOLD = 0.35
TOP_K = 3
DEBUG = False

USE_EXTRACTIVE_FALLBACK = True
EXTRACTIVE_SCORE_THRESHOLD = 0.50
MAX_EXTRACTIVE_CHARS = 500

_llm = None


def get_llm():
    global _llm

    if _llm is None:
        _llm = LocalLLM()

    return _llm


def print_sources(chunks):
    print("\nKaynaklar:")

    for chunk in chunks:
        print(
            f"- {chunk['source_name']} | "
            f"chunk_id={chunk['id']} | "
            f"score={chunk['score']:.4f}"
        )

def print_debug_info(question, chunks, messages):
    print("\n--- DEBUG MODE ---")

    print("\nKullanıcı sorusu:")
    print(question)

    print("\nRetrieved chunks:")
    for chunk in chunks:
        print("----")
        print(f"Chunk ID: {chunk['id']}")
        print(f"Kaynak: {chunk['source_name']}")
        print(f"Skor: {chunk['score']:.4f}")
        print("Metin:")
        print(chunk["chunk_text"])

    print("\nModele gönderilen mesajlar:")
    for message in messages:
        print("----")
        print("Role:", message["role"])
        print(message["content"])

    print("--- DEBUG END ---")


def should_use_extractive_answer(context_chunks):
    if not USE_EXTRACTIVE_FALLBACK:
        return False

    if len(context_chunks) != 1:
        return False

    best_chunk = context_chunks[0]

    if best_chunk["score"] < EXTRACTIVE_SCORE_THRESHOLD:
        return False

    if len(best_chunk["chunk_text"]) > MAX_EXTRACTIVE_CHARS:
        return False

    return True


def main():
    while True:
        question = input("\nSoru sor: ")

        if question.lower() in ["q", "quit", "exit"]:
            break
        
        if question.lower() == "/reindex":
          print("\nRe-index başlatılıyor...")
          ingest_documents()
          print("Re-index tamamlandı.")
          continue
        
        total_start_time = time.perf_counter()

        retrieval_start_time = time.perf_counter()
        chunks = get_top_chunks(question, top_k=TOP_K)
        retrieval_end_time = time.perf_counter()

        retrieval_time = retrieval_end_time - retrieval_start_time

        if not chunks:
            print("Veritabanında hiç chunk yok. Önce ingestion çalıştırılmalı.")
            continue

        best_score = chunks[0]["score"]

        print("\nBulunan en iyi skor:", f"{best_score:.4f}")

        if best_score < SIMILARITY_THRESHOLD:
            total_end_time = time.perf_counter()
            total_time = total_end_time - total_start_time

            print("\nCevap:")
            print("Bu bilgi verilen dokümanlarda yok.")

            print("\nPerformans:")
            print(f"- retrieval: {retrieval_time:.3f} saniye")
            print("- generation: 0.000 saniye")
            print(f"- toplam: {total_time:.3f} saniye")
            continue
        
        context_chunks = [chunk for chunk in chunks if chunk["score"] >= CONTEXT_SCORE_THRESHOLD]
        if not context_chunks:
            context_chunks = [chunks[0]]
        messages = build_rag_messages(question, context_chunks)

        if DEBUG:
            print_debug_info(question, context_chunks, messages)

        if should_use_extractive_answer(context_chunks):
            generation_start_time = time.perf_counter()
            answer = context_chunks[0]["chunk_text"]
            answer_mode = "extractive"
            generation_end_time = time.perf_counter()
        else:
            generation_start_time = time.perf_counter()
            answer = get_llm().generate_answer(messages)
            answer_mode = "generative"
            generation_end_time = time.perf_counter()

        generation_time = generation_end_time - generation_start_time

        total_end_time = time.perf_counter()
        total_time = total_end_time - total_start_time

        print("\nCevap:")
        print(answer.strip())

        print(f"\nCevap modu: {answer_mode}")

        print_sources(context_chunks)

        print("\nPerformans:")
        print(f"- retrieval: {retrieval_time:.3f} saniye")
        print(f"- generation: {generation_time:.3f} saniye")
        print(f"- toplam: {total_time:.3f} saniye")


if __name__ == "__main__":
    main()
