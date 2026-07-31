import logging
import re
from pathlib import Path

from pypdf import PdfReader

from app.database import init_db, replace_chunks
from app.embeddings import embed_texts, get_embedding_tokenizer
from app.index_state import build_source_manifest, list_document_paths


DOCS_DIR = Path("docs")

# Embedding modelinin sert sınırı 128 tokendır ve bu bütçe özel tokenları da
# kapsar; `split_long_text` sınırı hiçbir zaman aşmaz, bu yüzden 128 tam sınıra
# oturur ve kırpılma olmaz. Başka bir embedding modeline geçilirse bu değer
# yeniden kontrol edilmelidir.
#
# `tools/chunking_analysis.py`, 22 etiketli vaka (indekse dokunmadan ölçüldü):
#
#   boyut/overlap  chunk   R@1      R@3      R@5      MRR
#   60/12             77   0.5000   0.6364   0.6364   0.5530
#   80/16             53   0.6818   0.7727   0.7727   0.7197
#   110/20            47   0.8636   0.9773   1.0000   0.9318
#   120/20            43   0.9318   1.0000   1.0000   0.9773
#   128/12,20,30      38   0.9773   1.0000   1.0000   1.0000
#
# İki okuma notu:
#   - 60 ve 80 satırları retrieval kalitesi değildir. `R@5` bile 1.0'ın altında
#     çünkü eval imzaları "bu terimlerin hepsi aynı chunk'ta" der; küçük chunk
#     cevabı bölünce imza karşılanamaz hale gelir. Ölçüm yönteminin yanlılığıdır.
#     110/120/128'de `R@5 = 1.0`, yani aradaki fark gerçek sıralama farkıdır.
#   - Overlap'in ölçülebilir etkisi yok (12/20/30 birebir aynı). Bölme paragraf
#     bazlı olduğu için overlap nadiren devreye giriyor. 20 korundu; ölçümde
#     ayırt edilemeyen bir parametreyi değiştirmek gürültüye uymaktır.
CHUNK_SIZE = 128
CHUNK_OVERLAP = 20
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

    for file_path in list_document_paths(DOCS_DIR):
        if file_path.suffix.lower() == ".txt":
            documents.append(read_txt_file(file_path))
        elif file_path.suffix.lower() == ".pdf":
            documents.extend(read_pdf_file(file_path))

    return documents


def split_text_into_chunks(text, tokenizer=None):
    active_tokenizer = tokenizer or get_embedding_tokenizer()
    paragraphs = text.split("\n\n")

    chunks = []

    for paragraph in paragraphs:
        clean_paragraph = paragraph.strip()

        if clean_paragraph:
            chunks.extend(
                split_long_text(clean_paragraph, tokenizer=active_tokenizer)
            )

    return chunks


def split_long_text(
    text,
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    tokenizer=None,
):
    clean_text = " ".join(text.split())

    if not clean_text:
        return []

    active_tokenizer = tokenizer or get_embedding_tokenizer()
    special_token_count = active_tokenizer.num_special_tokens_to_add(pair=False)
    content_limit = chunk_size - special_token_count

    if content_limit < 1:
        raise ValueError("Chunk boyutu modelin özel token sayısından büyük olmalıdır.")

    if chunk_overlap < 0 or chunk_overlap >= content_limit:
        raise ValueError("Chunk overlap, kullanılabilir token sınırından küçük olmalıdır.")

    encoded = active_tokenizer(
        clean_text,
        add_special_tokens=False,
        return_offsets_mapping=True,
        truncation=False,
        verbose=False,
    )
    offsets = [
        tuple(offset)
        for offset in encoded["offset_mapping"]
        if offset[1] > offset[0]
    ]

    if len(offsets) <= content_limit:
        return [clean_text]

    chunks = []
    start_token = 0

    while start_token < len(offsets):
        hard_end_token = min(start_token + content_limit, len(offsets))
        end_token = hard_end_token

        if hard_end_token < len(offsets):
            midpoint_token = min(
                start_token + max(1, content_limit // 3),
                hard_end_token - 1,
            )
            search_start = offsets[midpoint_token][0]
            search_end = offsets[hard_end_token - 1][1]
            sentence_ends = list(
                SENTENCE_END_PATTERN.finditer(clean_text, search_start, search_end)
            )

            if sentence_ends:
                sentence_end = sentence_ends[-1].end()
                aligned_end = hard_end_token
                while (
                    aligned_end > start_token + 1
                    and offsets[aligned_end - 1][1] > sentence_end
                ):
                    aligned_end -= 1
                end_token = aligned_end
            else:
                end_token = align_token_end(
                    clean_text,
                    offsets,
                    start_token,
                    hard_end_token,
                )

        start_char = offsets[start_token][0]
        end_char = offsets[end_token - 1][1]
        chunk = clean_text[start_char:end_char].strip()

        if chunk:
            chunks.append(chunk)

        if end_token >= len(offsets):
            break

        desired_start = max(end_token - chunk_overlap, start_token + 1)
        next_start = align_token_start(
            clean_text,
            offsets,
            desired_start,
            end_token,
        )
        start_token = max(next_start, start_token + 1)

    return chunks


def align_token_end(text, offsets, start_token, hard_end_token):
    minimum_end = start_token + max(1, (hard_end_token - start_token) // 2)
    end_token = hard_end_token

    while end_token > minimum_end:
        end_char = offsets[end_token - 1][1]
        if end_char >= len(text) or text[end_char].isspace():
            break
        end_token -= 1

    return max(end_token, start_token + 1)


def align_token_start(text, offsets, desired_start, previous_end):
    previous_end_char = offsets[previous_end - 1][1]
    if text[previous_end_char - 1] in ".!?":
        return previous_end

    desired_char = offsets[desired_start][0]
    sentence_end = SENTENCE_END_PATTERN.search(
        text,
        desired_char,
        previous_end_char,
    )

    if sentence_end is not None:
        aligned_char = sentence_end.end()
        next_token = desired_start
        while (
            next_token < previous_end
            and offsets[next_token][0] < aligned_char
        ):
            next_token += 1

        if next_token < previous_end:
            return next_token

    next_token = desired_start
    while next_token < previous_end:
        start_char = offsets[next_token][0]
        if start_char == 0 or not text[start_char - 1].isalnum():
            return next_token
        next_token += 1

    return desired_start


def ingest_documents():
    init_db()
    source_manifest = build_source_manifest(DOCS_DIR)
    documents = read_documents()
    tokenizer = get_embedding_tokenizer()
    indexed_chunks = []

    for document in documents:
        chunks = split_text_into_chunks(document["text"], tokenizer=tokenizer)

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

    final_manifest = build_source_manifest(DOCS_DIR)

    if final_manifest != source_manifest:
        raise RuntimeError(
            "Dokümanlar indeksleme sırasında değişti; mevcut indeks korundu."
        )

    replace_chunks(indexed_chunks, source_manifest=source_manifest)
    return len(indexed_chunks)
