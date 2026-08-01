"""Üç elle seçilmiş eşiği ölçer: SIMILARITY_THRESHOLD, CONTEXT_SCORE_THRESHOLD,
EXTRACTIVE_SCORE_THRESHOLD. (Keşif aracı)

Bu dosya uygulamanın parçası DEĞİLDİR. `main.py` ve `app/` bunu içe aktarmaz;
eval de çalıştırmaz. İndekse dokunmaz, mevcut indeksten okur.

Bağlam: kanıt kapısı sorudan cevaba taşındıktan sonra (`app/groundedness.py`)
bu üç eşiğin taşıdığı yük değişti. Eskiden yanlış pozitifi tek başına bunlar
durduruyordu; artık groundedness de var. Bu araç onları VARSAYMADAN ölçer.

Çalıştırma (repository kökünden):

    source .venv/bin/activate
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python tools/threshold_analysis.py

Üç bölüm:

1. SIMILARITY_THRESHOLD — retrieval kapısı: `gate_score(chunks)` (en yüksek
   cosine) ALAKALI vakalarda ne kadar düşük, `not_found` vakalarında ne kadar
   yüksek olabiliyor. Boşluk = alakalı min - not_found max.
2. CONTEXT_SCORE_THRESHOLD — ikinci ve sonraki sıraların context'e girme
   eşiği. ALAKALI vakalarda 2+. sıradaki chunk'lar iki gruba ayrılır: chunk
   metni vakanın içerik imzasını karşılıyorsa MEŞRU (context'e girmeli),
   karşılamıyorsa GÜRÜLTÜ (girmemeli). Eşik bu iki grubu ayırmalı.
3. EXTRACTIVE_SCORE_THRESHOLD — extractive kısayolunun `best_source.score`
   eşiği. Yalnızca context'e TEK chunk giren vakalarla ölçülür, çünkü kısayol
   `len(sources) == 1` şartına bağlı. ALAKALI vakalarda skor yüksek olmalı
   (kısayol güvenle tetiklenir), `not_found`/tuzak vakalarda düşük olmalı
   (zaten SIMILARITY_THRESHOLD'u geçmiş olsalar bile kısayol tetiklenmemeli).

Ayrıntı ve sonuçların yorumu için `kalibrasyon-kaydi` skill'i.
"""

import json
from pathlib import Path

from app.config import (
    CONTEXT_SCORE_THRESHOLD,
    EXTRACTIVE_SCORE_THRESHOLD,
    SIMILARITY_THRESHOLD,
    TOP_K,
)
from app.eval_metrics import chunk_matches_signature
from app.rag_service import RAGService
from app.retrieval import gate_score, get_top_chunks


def load_cases():
    return json.loads(Path("eval_cases.json").read_text(encoding="utf-8"))


def collect_rows():
    cases = load_cases()
    service = RAGService()
    rows = []

    for case in cases:
        chunks = get_top_chunks(case["question"], top_k=TOP_K)
        best_score = gate_score(chunks)
        signatures = case.get("relevant_chunk_terms") or []

        secondary = []
        for chunk in chunks[1:]:
            legit = any(
                chunk_matches_signature(chunk["chunk_text"], signature)
                for signature in signatures
            )
            secondary.append({"score": chunk["score"], "legit": legit})

        # Extractive kısayolu `select_matched_context_chunks()`'ın kaç chunk
        # döndürdüğüne bağlı; TOP_K=3 döndüğü için ham chunk sayısı değil,
        # mevcut kapının (CONTEXT_SCORE_THRESHOLD + göreli marj) seçtiği
        # context kullanılmalı.
        matched = service.select_matched_context_chunks(chunks, question=case["question"])

        rows.append({
            "name": case["name"],
            "expectation": case["expectation"],
            "best_score": best_score,
            "chunk0_score": chunks[0]["score"] if chunks else 0.0,
            "secondary": secondary,
            "single_source": len(matched) == 1,
        })

    return rows


def sweep(relevant_values, trap_values, candidates):
    print(f"{'eşik':>8}{'alakalı geçen':>16}{'tuzak geçen':>14}")
    for value in candidates:
        rel_pass = sum(1 for v in relevant_values if v >= value) / len(relevant_values)
        trap_pass = sum(1 for v in trap_values if v >= value) / len(trap_values)
        print(f"{value:>8.3f}{rel_pass:>15.1%}{trap_pass:>14.1%}")


def report_separation(label, relevant_values, trap_values):
    if not relevant_values or not trap_values:
        print(f"\n{label}: bir grup boş, ölçülemedi.")
        return None

    rel_min = min(relevant_values)
    trap_max = max(trap_values)
    gap = rel_min - trap_max
    print(f"\n{label}")
    print(f"  alakalı min = {rel_min:.4f}  (n={len(relevant_values)})")
    print(f"  tuzak max   = {trap_max:.4f}  (n={len(trap_values)})")
    print(f"  boşluk      = {gap:+.4f}")
    if gap > 0:
        print(f"  güvenli aralık: {trap_max:.4f} - {rel_min:.4f}")
    else:
        print("  AYIRT ETMİYOR: hiçbir tek eşik iki grubu ayıramaz.")
    return rel_min, trap_max, gap


def section_similarity(rows):
    print("\n" + "=" * 70)
    print("1. SIMILARITY_THRESHOLD (retrieval kapısı, mevcut = "
          f"{SIMILARITY_THRESHOLD})")
    print("=" * 70)

    relevant = [r["best_score"] for r in rows if r["expectation"] == "relevant"]
    traps = [r["best_score"] for r in rows if r["expectation"] == "not_found"]

    report_separation("gate_score(chunks) ayrımı", relevant, traps)

    print()
    candidates = [round(0.10 + 0.02 * i, 2) for i in range(16)]
    sweep(relevant, traps, candidates)


def section_context(rows):
    print("\n" + "=" * 70)
    print("2. CONTEXT_SCORE_THRESHOLD (mevcut = "
          f"{CONTEXT_SCORE_THRESHOLD})")
    print("=" * 70)

    legit = []
    noise = []
    for row in rows:
        if row["expectation"] != "relevant":
            continue
        for entry in row["secondary"]:
            (legit if entry["legit"] else noise).append(entry["score"])

    print(f"2+. sıradaki chunk sayısı: meşru={len(legit)}, gürültü={len(noise)}")
    result = report_separation("2+. sıra cosine ayrımı (meşru vs gürültü)", legit, noise)

    print()
    candidates = [round(0.10 + 0.02 * i, 2) for i in range(16)]
    print(f"{'eşik':>8}{'meşru geçen':>14}{'gürültü geçen':>16}")
    for value in candidates:
        legit_pass = sum(1 for v in legit if v >= value) / len(legit) if legit else float("nan")
        noise_pass = sum(1 for v in noise if v >= value) / len(noise) if noise else float("nan")
        print(f"{value:>8.3f}{legit_pass:>13.1%}{noise_pass:>16.1%}")

    return result


def section_extractive(rows):
    print("\n" + "=" * 70)
    print("3. EXTRACTIVE_SCORE_THRESHOLD (mevcut = "
          f"{EXTRACTIVE_SCORE_THRESHOLD})")
    print("=" * 70)

    single_source = [row for row in rows if row["single_source"] and row["best_score"] >= SIMILARITY_THRESHOLD]
    relevant = [row["chunk0_score"] for row in single_source if row["expectation"] == "relevant"]
    traps = [row["chunk0_score"] for row in single_source if row["expectation"] == "not_found"]

    print(f"Tek kaynaklı vaka sayısı: alakalı={len(relevant)}, not_found={len(traps)}")
    report_separation("chunk[0].score ayrımı (tek kaynaklı vakalarda)", relevant, traps)

    print()
    candidates = [round(0.10 + 0.02 * i, 2) for i in range(21)]
    sweep(relevant, traps, candidates) if relevant and traps else None


def main():
    rows = collect_rows()
    section_similarity(rows)
    section_context(rows)
    section_extractive(rows)


if __name__ == "__main__":
    main()
