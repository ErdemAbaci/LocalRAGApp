"""Retrieval metriklerinin saf hesaplaması.

Bu modül veritabanı, embedding veya dosya sistemi bilmez. Girdi olarak
retrieval sonuçlarını ve etiketli imzaları alır, çıktı olarak Recall@k ve MRR
üretir. Böylece metrik mantığı gerçek model yüklemeden test edilebilir.

Ground truth chunk ID ile değil **içerik imzası** ile etiketlenir. Bir imza,
doğru chunk'ı benzersiz kılan terimlerin listesidir; bir chunk o terimlerin
hepsini içeriyorsa imzayı karşılar. Chunk ID'leri her reindex'te değiştiği ve
chunking ayarı değişince chunk sınırları da kaydığı için ID bazlı etiketler
kısa sürede geçersiz olurdu.
"""


# Python'un casefold'u Türkçe'yi doğru küçültmez: "İ".casefold() sonucu "i"
# değil, "i" + U+0307 (birleşen nokta) olur ve "I".casefold() "ı" yerine "i"
# verir. Bu yüzden büyük harfle yazılmış bir imza sessizce eşleşmezdi.
# Türkçe eşlemeyi önce elle yapıp artakalan birleşen noktayı temizliyoruz.
TURKISH_LOWER_MAP = str.maketrans({"İ": "i", "I": "ı"})
COMBINING_DOT_ABOVE = "̇"


def normalize_text(text):
    lowered = str(text).translate(TURKISH_LOWER_MAP).casefold()
    lowered = lowered.replace(COMBINING_DOT_ABOVE, "")

    return " ".join(lowered.split())


def chunk_matches_signature(chunk_text, signature):
    normalized_chunk = normalize_text(chunk_text)

    return all(
        normalize_text(term) in normalized_chunk
        for term in signature
    )


def find_signature_ranks(results, signatures):
    """Her imza için onu karşılayan en iyi sıralamayı (1 tabanlı) döndürür.

    İmza hiçbir sonuçta bulunamazsa değeri None olur.
    """
    ranks = []

    for signature in signatures:
        matched_rank = None

        for position, result in enumerate(results, start=1):
            if chunk_matches_signature(result["chunk_text"], signature):
                matched_rank = position
                break

        ranks.append(matched_rank)

    return ranks


def recall_at_k(signature_ranks, k):
    """İlk k sonuçta bulunan imzaların oranı."""
    if not signature_ranks:
        return None

    found = sum(
        1
        for rank in signature_ranks
        if rank is not None and rank <= k
    )

    return found / len(signature_ranks)


def reciprocal_rank(signature_ranks):
    """İlk doğru sonucun sırasının tersi. Hiç bulunamazsa 0."""
    found_ranks = [rank for rank in signature_ranks if rank is not None]

    if not found_ranks:
        return 0.0

    return 1.0 / min(found_ranks)


def average(values):
    numeric_values = [value for value in values if value is not None]

    if not numeric_values:
        return None

    return sum(numeric_values) / len(numeric_values)


def summarize_case_metrics(case_metrics, k_values=(1, 3, 5)):
    """Vaka bazlı metrikleri tek bir özet sözlüğüne indirger."""
    summary = {"case_count": len(case_metrics)}

    for k in k_values:
        summary[f"recall_at_{k}"] = average(
            [metrics.get(f"recall_at_{k}") for metrics in case_metrics]
        )

    summary["mrr"] = average(
        [metrics.get("reciprocal_rank") for metrics in case_metrics]
    )

    return summary


def build_case_metrics(results, signatures, k_values=(1, 3, 5)):
    signature_ranks = find_signature_ranks(results, signatures)
    metrics = {
        "signature_ranks": signature_ranks,
        "reciprocal_rank": reciprocal_rank(signature_ranks),
    }

    for k in k_values:
        metrics[f"recall_at_{k}"] = recall_at_k(signature_ranks, k)

    return metrics


def find_unmatched_signatures(all_chunks, signatures):
    """İndeksin tamamında hiçbir chunk'ın karşılamadığı imzaları döndürür.

    İmza bazlı etiketlemenin tek gerçek riski budur: yanlış yazılmış bir imza
    sessizce "hiç bulunamadı" sayılır ve metrikleri haksız yere düşürür. Bu
    kontrol, etiketin kendisinin bozuk olduğunu ayrı bir hata olarak gösterir.
    """
    unmatched = []

    for signature in signatures:
        matched = any(
            chunk_matches_signature(chunk["chunk_text"], signature)
            for chunk in all_chunks
        )

        if not matched:
            unmatched.append(signature)

    return unmatched


def format_metric(value):
    if value is None:
        return "-"

    return f"{value:.4f}"


def compare_summaries(baseline, current, k_values=(1, 3, 5)):
    """Baseline ile güncel özeti karşılaştırıp metrik başına fark üretir."""
    metric_names = [f"recall_at_{k}" for k in k_values] + ["mrr"]
    comparisons = []

    for name in metric_names:
        baseline_value = baseline.get(name)
        current_value = current.get(name)

        if baseline_value is None or current_value is None:
            delta = None
        else:
            delta = current_value - baseline_value

        comparisons.append({
            "name": name,
            "baseline": baseline_value,
            "current": current_value,
            "delta": delta,
        })

    return comparisons
