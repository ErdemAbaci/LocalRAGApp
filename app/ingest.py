import logging
import re
from pathlib import Path

from pypdf import PdfReader

from app.database import init_db, replace_chunks
from app.embeddings import embed_texts


DOCS_DIR = Path("docs")
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
SENTENCE_END_PATTERN = re.compile(r"[.!?](?=\s|$)")


class IgnoredPdfObjectFilter(logging.Filter):
    def filter(self, record):
        return not record.getMessage().startswith("Ignoring wrong pointing object")


logging.getLogger("pypdf._reader").addFilter(IgnoredPdfObjectFilter())


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
            search_start = start + chunk_size // 2
            sentence_ends = list(SENTENCE_END_PATTERN.finditer(clean_text, search_start, end))

            if sentence_ends:
                end = sentence_ends[-1].end()
            else:
                split_at = clean_text.rfind(" ", search_start, end)

                if split_at != -1:
                    end = split_at

        chunk = clean_text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(clean_text):
            break

        desired_start = max(end - chunk_overlap, start + 1)
        next_start = align_chunk_start(clean_text, desired_start, end)
        start = max(next_start, start + 1)

    return chunks


def align_chunk_start(text, desired_start, previous_end):
    if desired_start <= 0:
        return 0

    sentence_ends = list(SENTENCE_END_PATTERN.finditer(text, desired_start, previous_end))

    if sentence_ends:
        start = sentence_ends[0].end()

        while start < len(text) and text[start].isspace():
            start += 1

        if start < previous_end:
            return start

    if text[previous_end - 1] in ".!?":
        start = previous_end

        while start < len(text) and text[start].isspace():
            start += 1

        return start

    next_space = text.find(" ", desired_start, previous_end)

    if next_space != -1:
        return next_space + 1

    previous_space = text.rfind(" ", 0, desired_start)

    if previous_space != -1:
        return previous_space + 1

    return desired_start


def ingest_documents():
    init_db()
    documents = read_documents()
    indexed_chunks = []

    for document in documents:
        chunks = split_text_into_chunks(document["text"])

        if not chunks:
            continue

        embeddings = embed_texts(chunks)

        for chunk_index, (chunk, embedding) in enumerate(zip(chunks, embeddings), start=1):
            indexed_chunks.append({
                "source_name": document["source_name"],
                "source_type": document["source_type"],
                "page_number": document["page_number"],
                "chunk_index": chunk_index,
                "chunk_text": chunk,
                "embedding": embedding,
            })

    if not indexed_chunks:
        raise ValueError("İndekslenecek metin bulunamadı; mevcut indeks korundu.")

    replace_chunks(indexed_chunks)
    return len(indexed_chunks)
