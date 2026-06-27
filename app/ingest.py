from pathlib import Path

from sentence_transformers import SentenceTransformer

from app.database import init_db, clear_chunks, insert_chunk


DOCS_DIR = Path("docs")
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def read_text_files():
    documents = []

    for file_path in DOCS_DIR.glob("*.txt"):
        text = file_path.read_text(encoding="utf-8")

        documents.append({
            "source_name": file_path.name,
            "text": text
        })

    return documents


def split_text_into_chunks(text):
    paragraphs = text.split("\n\n")

    chunks = []

    for paragraph in paragraphs:
        clean_paragraph = paragraph.strip()

        if clean_paragraph:
            chunks.append(clean_paragraph)

    return chunks


def ingest_documents():
    init_db()
    clear_chunks()

    model = SentenceTransformer(MODEL_NAME)

    documents = read_text_files()

    total_chunks = 0

    for document in documents:
        chunks = split_text_into_chunks(document["text"])

        for chunk in chunks:
            embedding = model.encode(chunk).tolist()

            insert_chunk(
                source_name=document["source_name"],
                chunk_text=chunk,
                embedding=embedding
            )

            total_chunks += 1

    print(f"Ingestion tamamlandı. Toplam chunk sayısı: {total_chunks}")