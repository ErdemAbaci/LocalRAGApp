from pathlib import Path

from pypdf import PdfReader

from app.database import init_db, clear_chunks, insert_chunk
from app.embeddings import embed_texts


DOCS_DIR = Path("docs")
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100


def read_txt_file(file_path):
    text = file_path.read_text(encoding="utf-8")

    return {
        "source_name": file_path.name,
        "source_type": "txt",
        "page_number": None,
        "text": text
    }


def read_pdf_file(file_path):
    reader = PdfReader(file_path)
    documents = []

    for page_index, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        clean_text = text.strip()

        if not clean_text:
            continue

        documents.append({
            "source_name": file_path.name,
            "source_type": "pdf",
            "page_number": page_index,
            "text": clean_text
        })

    return documents


def read_documents():
    documents = []

    for file_path in DOCS_DIR.glob("*.txt"):
        documents.append(read_txt_file(file_path))

    for file_path in DOCS_DIR.glob("*.pdf"):
        documents.extend(read_pdf_file(file_path))

    return documents


def split_text_into_chunks(text):
    paragraphs = text.split("\n\n")

    chunks = []

    for paragraph in paragraphs:
        clean_paragraph = paragraph.strip()

        if clean_paragraph:
            chunks.extend(split_long_text(clean_paragraph))

    return chunks


def split_long_text(text, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
    clean_text = " ".join(text.split())

    if len(clean_text) <= chunk_size:
        return [clean_text] if clean_text else []

    chunks = []
    start = 0

    while start < len(clean_text):
        end = min(start + chunk_size, len(clean_text))

        if end < len(clean_text):
            split_at = clean_text.rfind(" ", start + chunk_size // 2, end)

            if split_at != -1:
                end = split_at

        chunk = clean_text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(clean_text):
            break

        start = max(end - chunk_overlap, start + 1)

        while start < len(clean_text) and clean_text[start].isspace():
            start += 1

    return chunks


def ingest_documents():
    init_db()
    clear_chunks()

    documents = read_documents()

    total_chunks = 0

    for document in documents:
        chunks = split_text_into_chunks(document["text"])

        if not chunks:
            continue

        embeddings = embed_texts(chunks)

        for chunk_index, (chunk, embedding) in enumerate(zip(chunks, embeddings), start=1):
            insert_chunk(
                source_name=document["source_name"],
                source_type=document["source_type"],
                page_number=document["page_number"],
                chunk_index=chunk_index,
                chunk_text=chunk,
                embedding=embedding
            )

            total_chunks += 1

    print(f"Ingestion tamamlandı. Toplam chunk sayısı: {total_chunks}")
