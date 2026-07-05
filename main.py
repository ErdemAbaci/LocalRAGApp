import time

from app.database import get_chunk_stats
from app.embeddings import MODEL_NAME as EMBEDDING_MODEL_NAME
from app.retrieval import get_top_chunks
from app.prompts import build_rag_messages
from app.llm import LocalLLM, MODEL_ALIAS
from app.ingest import CHUNK_OVERLAP, CHUNK_SIZE, ingest_documents

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


def print_banner():
    print()
    print("==================================================")
    print("Local RAG Assistant")
    print("Yerel doküman soru-cevap asistanı")
    print("==================================================")
    print(f"Embedding: {EMBEDDING_MODEL_NAME}")
    print(f"LLM: {MODEL_ALIAS} (gerektiğinde yüklenir)")
    print("Docs: docs/")
    print("Database: data/rag.db")
    print()
    print("Komutlar: /help, /stats, /reindex, /debug on, /debug off, /exit")


def print_help():
    print("\nKomutlar:")
    print("- /help       Komut listesini gösterir")
    print("- /stats      Index ve model bilgilerini gösterir")
    print("- /reindex    docs/ klasörünü yeniden indexler")
    print("- /debug on   Debug çıktısını açar")
    print("- /debug off  Debug çıktısını kapatır")
    print("- /exit       Uygulamadan çıkar")
    print("\nNormal soru sormak için doğrudan yazman yeterli.")


def print_stats():
    stats = get_chunk_stats()

    print("\nSistem durumu:")
    print(f"- chunk sayısı: {stats['total_chunks']}")
    print(f"- kaynak dosya sayısı: {stats['source_count']}")
    print(f"- veritabanı: {stats['db_path']}")
    print(f"- embedding modeli: {EMBEDDING_MODEL_NAME}")
    print(f"- llm modeli: {MODEL_ALIAS}")
    print(f"- debug: {'açık' if DEBUG else 'kapalı'}")
    print(f"- top_k: {TOP_K}")
    print(f"- chunk size: {CHUNK_SIZE}")
    print(f"- chunk overlap: {CHUNK_OVERLAP}")
    print(f"- similarity threshold: {SIMILARITY_THRESHOLD}")
    print(f"- context score threshold: {CONTEXT_SCORE_THRESHOLD}")


def print_sources(chunks):
    print("\nKaynaklar:")

    for chunk in chunks:
        source_parts = [chunk["source_name"]]

        if chunk.get("page_number") is not None:
            source_parts.append(f"page={chunk['page_number']}")

        if chunk.get("chunk_index") is not None:
            source_parts.append(f"chunk_index={chunk['chunk_index']}")

        source_parts.append(f"chunk_id={chunk['id']}")
        source_parts.append(f"score={chunk['score']:.4f}")

        print("- " + " | ".join(source_parts))

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


def handle_command(command):
    global DEBUG

    if command in ["/exit", "/quit", "q", "quit", "exit"]:
        return "exit"

    if command == "/help":
        print_help()
        return "handled"

    if command == "/stats":
        print_stats()
        return "handled"

    if command == "/reindex":
        print("\nRe-index başlatılıyor...")
        ingest_documents()
        print("Re-index tamamlandı.")
        return "handled"

    if command == "/debug on":
        DEBUG = True
        print("\nDebug modu açıldı.")
        return "handled"

    if command == "/debug off":
        DEBUG = False
        print("\nDebug modu kapatıldı.")
        return "handled"

    if command.startswith("/"):
        print("\nBilinmeyen komut. Komutları görmek için /help yaz.")
        return "handled"

    return None


def main():
    print_banner()

    while True:
        question = input("\nrag> ").strip()

        if not question:
            continue

        command_result = handle_command(question.lower())

        if command_result == "exit":
            break

        if command_result == "handled":
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
