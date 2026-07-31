"""Chunking ayarlarını gerçek eval seti üzerinde karşılaştırır. (Keşif aracı)

Bu dosya uygulamanın parçası DEĞİLDİR ve **indekse dokunmaz**. Her ayar için
dokümanları yeniden parçalar, embeddingleri bellekte üretir ve `rank_chunks()`
ile aynı sıralama mantığını çalıştırır. `data/rag.db` değişmez; ölçüm bittikten
sonra `/reindex` gerekmez.

Cevaplamaya çalıştığı soru: `CHUNK_SIZE = 110` ve `CHUNK_OVERLAP = 20` sezgiyle
seçilmişti; başka bir ayar aynı sorularda daha iyi sıralama üretiyor mu?

Neden chunking sıralamayı etkiler? İki karşıt baskı var:

- **Küçük chunk**: içindeki her kelime konuyla ilgili olduğu için embedding
  keskinleşir ve eşleşme netleşir. Ama cevap iki parçaya bölünürse tek bir
  chunk sorunun tamamını karşılayamaz.
- **Büyük chunk**: cevabın tamamını içerme şansı yükselir. Ama embedding tek bir
  384 boyutlu vektöre sıkıştığı için birbiriyle ilgisiz cümleler ortalanır ve
  ayırt edicilik düşer.

Sert sınır: embedding modeli 128 token alır. `CHUNK_SIZE` bunun üstüne
çıkarılırsa metnin kuyruğu sessizce gömülmeden kalır, bu yüzden 128 üstü ayar
ölçülmez.

Çalıştırma (repository kökünden):

    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python tools/chunking_analysis.py
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.embeddings import embed_texts, get_embedding_tokenizer  # noqa: E402
from app.eval_metrics import (  # noqa: E402
    build_case_metrics,
    format_metric,
    summarize_case_metrics,
)
from app.ingest import read_documents, split_long_text  # noqa: E402
from app.retrieval import rank_chunks  # noqa: E402

CASES_PATH = "eval_cases.json"
METRIC_TOP_K = 5
METRIC_K_VALUES = (1, 3, 5)

# (chunk_size, chunk_overlap). Mevcut ayar 110/20. Overlap oranları kabaca
# %10, %18 ve %27; boyutlar 128 token sınırının altında tutuldu.
CONFIGURATIONS = (
    (60, 12),
    (80, 16),
    (110, 20),
    (120, 20),
    (128, 12),
    (128, 20),
    (128, 30),
)


def load_labeled_cases():
    with open(CASES_PATH, "r", encoding="utf-8") as cases_file:
        cases = json.load(cases_file)

    return [case for case in cases if case.get("relevant_chunk_terms")]


def build_chunks(chunk_size, chunk_overlap):
    """Dokümanları verilen ayarla parçalar ve embeddingleriyle döndürür.

    `app.ingest.split_long_text` doğrudan çağrılır; araç kendi parçalama
    kopyasını tutmaz, yoksa ölçüm uygulamanın gerçek davranışını yansıtmazdı.
    """
    tokenizer = get_embedding_tokenizer()
    records = []

    for document in read_documents():
        chunk_index = 0

        for paragraph in document["text"].split("\n\n"):
            clean_paragraph = paragraph.strip()

            if not clean_paragraph:
                continue

            for text in split_long_text(
                clean_paragraph,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                tokenizer=tokenizer,
            ):
                chunk_index += 1
                records.append({
                    "id": len(records) + 1,
                    "source_name": document["source_name"],
                    "source_type": document.get("source_type"),
                    "page_number": document.get("page_number"),
                    "chunk_index": chunk_index,
                    "chunk_text": text,
                })

    embeddings = embed_texts([record["chunk_text"] for record in records])

    for record, embedding in zip(records, embeddings):
        record["embedding"] = embedding

    return records


def measure(cases, chunks):
    case_metrics = []

    for case in cases:
        results = rank_chunks(case["question"], chunks, top_k=METRIC_TOP_K)
        case_metrics.append(build_case_metrics(
            results,
            case["relevant_chunk_terms"],
            k_values=METRIC_K_VALUES,
        ))

    return summarize_case_metrics(case_metrics, k_values=METRIC_K_VALUES)


def chunk_length_stats(chunks):
    tokenizer = get_embedding_tokenizer()
    lengths = [
        len(tokenizer(chunk["chunk_text"])["input_ids"])
        for chunk in chunks
    ]

    return sum(lengths) / len(lengths), max(lengths)


def main():
    cases = load_labeled_cases()
    print(f"{len(cases)} etiketli vaka, {len(CONFIGURATIONS)} ayar\n")

    header = (
        f"{'boyut/overlap':<15}{'chunk':>7}{'ort.token':>11}{'maks':>6}"
        f"{'R@1':>9}{'R@3':>9}{'R@5':>9}{'MRR':>9}"
    )
    print(header)
    print("-" * len(header))

    rows = []

    for chunk_size, chunk_overlap in CONFIGURATIONS:
        chunks = build_chunks(chunk_size, chunk_overlap)
        summary = measure(cases, chunks)
        average_length, longest = chunk_length_stats(chunks)
        rows.append(((chunk_size, chunk_overlap), summary))

        print(
            f"{f'{chunk_size}/{chunk_overlap}':<15}{len(chunks):>7}"
            f"{average_length:>11.1f}{longest:>6}"
            f"{format_metric(summary['recall_at_1']):>9}"
            f"{format_metric(summary['recall_at_3']):>9}"
            f"{format_metric(summary['recall_at_5']):>9}"
            f"{format_metric(summary['mrr']):>9}"
        )

    best = max(rows, key=lambda row: (row[1]["mrr"], row[1]["recall_at_1"]))
    print(f"\nMRR'a göre en iyi ayar: {best[0][0]}/{best[0][1]}")
    print(
        "Fark küçükse mevcut ayarda kal; ölçümde ayırt edilemeyen bir "
        "parametreyi değiştirmek gürültüye uymaktır."
    )


if __name__ == "__main__":
    main()
