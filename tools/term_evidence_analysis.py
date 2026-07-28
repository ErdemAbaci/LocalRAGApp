"""Kelime kanıtı sinyalinin ayırt ediciliğini ölçer. (Keşif aracı)

Bu dosya uygulamanın parçası DEĞİLDİR. `main.py` ve `app/` bunu içe aktarmaz;
eval de çalıştırmaz. Amacı bir tasarım kararını veriyle beslemektir ve tek
başına çalıştırılır.

Cevaplamaya çalıştığı soru: "sorunun kelimeleri modele giden metinde gerçekten
geçiyor mu" sinyali, cevabı dokümanda bulunan soruları bulunmayanlardan
ayırabiliyor mu?

Çalıştırma (repository kökünden):

    source .venv/bin/activate
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python tools/term_evidence_analysis.py

Çıktı üç bölümdür: vaka başına kapsama tablosu, kelime bazında eşleşen/eksik
detayı ve grup özeti.

2026-07 ölçümünün sonucu (17 eval sorusu, 24 chunk):

- Tam eşleşmede alakalı sorular 0.71-1.00, hard negative'ler 0.00-0.33 aldı.
  Aradaki boşluk nettir; cosine skorunda ise iki grup örtüşüyordu.
- Kelimenin ilk 4-5 harfine bakan kaba kök alma sinyali BOZDU: hard negative
  en yüksek değeri 0.75'e çıktı ve boşluk 0.38'den 0.05'e düştü. Sebep yanlış
  eşleşmelerdir (`sayısı` -> `sayısal`, `yapılandırılmalıdır` -> `yapılır`).
  Bu iş için kesinlik, kapsayıcılıktan önemlidir.

Ayrıntı ve sonuçların yorumu için AGENTS.md bölüm 7.
"""

import json
import re
from pathlib import Path

from app.eval_metrics import normalize_text
from app.rag_service import RAGService
from app.retrieval import get_top_chunks
from app.config import TOP_K


# Soru kalıbı ve genel dilbilgisi kelimeleri. Bunlar her soruda geçtiği için
# ayırt edici değildir; sinyale dahil edilirse hard negative'ler de yüksek
# kapsama alır ve ölçüm anlamsızlaşır.
# DİKKAT: bu liste normalize_text() çıktısıyla karşılaştırılır; normalize_text
# Türkçe karakterleri korur (yalnızca İ/I eşlemesi yapar). Bu yüzden stopword'ler
# de Türkçe karakterleriyle yazılmalıdır. ASCII yazmak listeyi sessizce etkisiz
# bırakır.
STOPWORDS = {
    "ne", "nedir", "nasıl", "neden", "niçin", "hangi", "kaç", "kim", "nerede",
    "nelerdir", "midir", "mi", "mu", "mü", "mı", "ve", "veya", "ile", "için",
    "bir", "bu", "şu", "da", "de", "ki", "en", "çok", "az", "olan", "olarak",
    "gibi", "sonra", "önce", "üzere", "göre", "kadar", "daha", "ise", "ama",
    "fakat", "ancak", "yani", "eğer", "her", "hiç", "bazı", "tüm", "olmalıdır",
    "olmalı", "kullanılır", "yapılır", "edilir", "edilmelidir", "seçilir",
    "alınmalıdır", "yapılandırılmalıdır", "kullanılmalıdır", "önerir",
    "izlenir", "var", "yok", "eder", "olur", "pişirilir",
}

WORD_PATTERN = re.compile(r"[0-9a-zçğıöşü\-]+")


def content_words(question):
    normalized = normalize_text(question)
    tokens = WORD_PATTERN.findall(normalized)

    return [
        token
        for token in tokens
        if token not in STOPWORDS and len(token) >= 3
    ]


def build_context_text(results):
    service = RAGService()
    matched = service.select_matched_context_chunks(results)

    if not matched:
        return ""

    context_chunks = service.order_context_chunks(
        service.expand_context_chunks(matched),
        matched,
    )

    return normalize_text("\n".join(
        chunk["chunk_text"] for chunk in context_chunks
    ))


def coverage(words, context, prefix_length=None):
    if not words:
        return None, []

    matched = []
    for word in words:
        needle = word if prefix_length is None else word[:prefix_length]
        if needle and needle in context:
            matched.append(word)

    return len(matched) / len(words), matched


def main():
    cases = json.loads(
        Path("eval_cases.json").read_text(encoding="utf-8")
    )

    rows = []

    for case in cases:
        question = case["question"]
        results = get_top_chunks(question, top_k=TOP_K)
        context = build_context_text(results)
        words = content_words(question)

        exact_ratio, exact_matched = coverage(words, context)
        prefix5_ratio, prefix5_matched = coverage(words, context, prefix_length=5)
        prefix4_ratio, prefix4_matched = coverage(words, context, prefix_length=4)

        if case["expectation"] == "relevant":
            group = "ALAKALI"
        elif case.get("difficulty") == "hard":
            group = "TUZAK"
        else:
            group = "kolay-neg"

        rows.append({
            "group": group,
            "name": case["name"],
            "score": results[0]["score"] if results else 0.0,
            "words": words,
            "exact": exact_ratio,
            "exact_matched": exact_matched,
            "prefix5": prefix5_ratio,
            "prefix5_matched": prefix5_matched,
            "prefix4": prefix4_ratio,
        })

    print(f"{'GRUP':<10} {'VAKA':<36} {'SKOR':>6} {'TAM':>6} {'ÖN5':>6} {'ÖN4':>6}")
    print("-" * 76)

    for row in sorted(rows, key=lambda item: (item["group"], -item["score"])):
        def fmt(value):
            return "-" if value is None else f"{value:.2f}"

        print(
            f"{row['group']:<10} {row['name']:<36} "
            f"{row['score']:>6.4f} {fmt(row['exact']):>6} "
            f"{fmt(row['prefix5']):>6} {fmt(row['prefix4']):>6}"
        )

    print("\n\nKELİME DETAYI (tam eşleşme)\n")
    for row in sorted(rows, key=lambda item: (item["group"], -item["score"])):
        missing = [w for w in row["words"] if w not in row["exact_matched"]]
        print(f"[{row['group']}] {row['name']}")
        print(f"    kelimeler : {row['words']}")
        print(f"    eşleşen   : {row['exact_matched']}")
        print(f"    eksik     : {missing}")
        prefix_gain = [
            w for w in row["prefix5_matched"]
            if w not in row["exact_matched"]
        ]
        if prefix_gain:
            print(f"    ön5 ile kazanılan: {prefix_gain}")
        print()

    print("\nGRUP ÖZETİ\n")
    for group in ["ALAKALI", "TUZAK", "kolay-neg"]:
        group_rows = [r for r in rows if r["group"] == group]
        if not group_rows:
            continue

        for metric in ["exact", "prefix5", "prefix4"]:
            values = [r[metric] for r in group_rows if r[metric] is not None]
            if not values:
                continue
            print(
                f"  {group:<10} {metric:<8} "
                f"ort={sum(values)/len(values):.2f} "
                f"min={min(values):.2f} max={max(values):.2f}"
            )
        print()


if __name__ == "__main__":
    main()
