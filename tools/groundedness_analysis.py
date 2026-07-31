"""Groundedness sinyalinin ayırt ediciliğini ölçer. (Keşif aracı)

Bu dosya uygulamanın parçası DEĞİLDİR. `main.py` ve `app/` bunu içe aktarmaz;
eval de çalıştırmaz. Amacı bir eşik kararını veriyle beslemektir.

Çalıştırma (repository kökünden):

    source .venv/bin/activate
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python tools/groundedness_analysis.py

--- Ölçüm neyi taklit ediyor ---

Asıl ölçmek istediğimiz şey "modelin ürettiği cevap context'e dayanıyor mu".
Ama model çıktısı deterministik değildir ve her ölçümde LLM çalıştırmak eşik
kalibrasyonunu tekrarlanamaz kılar. Bu yüzden cevabın yerine üç vekil metin
grubu ölçülür:

- DAYANAKLI  : context'in kendi cümleleri. Üst sınır. Mekanizmanın Türkçe
               morfolojide bir şeyi kaçırmadığını doğrular; 1.00 beklenir.
- PARAFRAZ   : vakanın sorusu. Aynı içeriği **kullanıcının kelimeleriyle**
               anlatan bir metindir, yani sistematik olarak yeniden ifade
               edilmiş bir cevabın en kötü hâlidir. Gerçek cevap context'ten
               kopyalayarak yazıldığı için bundan daha yüksek skor alır.
               Meşru cevabın ALT SINIRI olarak okunmalıdır.
- DAYANAKSIZ : başka bir dokümandan alınmış cümleler. Uydurmanın vekilidir.

--- İki eşik iki ayrı ölçümle belirlenir ---

`GROUNDEDNESS_SENTENCE_SUPPORT` **cümle** seviyesinde ölçülür: bir cümlenin
kelimelerinin kaçta kaçı context'te geçiyor. Bu sürekli bir değerdir ve asıl
ayırt edici ölçüm budur.

`GROUNDEDNESS_THRESHOLD` **cevap** seviyesinde ölçülür: desteklenen cümlelerin
oranı. Tek cümlelik bir metinde bu oran yalnızca 0.00 veya 1.00 olabilir, bu
yüzden cevap seviyesi ölçümü çok cümleli vekillerle yapılmalıdır. İlk ölçümde
bu gözden kaçtı ve tek cümlelik vekiller "boşluk yok" gibi görünen bir sonuç
üretti; oysa cümle seviyesinde ayrım 0.98'e karşı 0.05'ti.

Boşluk cümle seviyesinde de negatifse groundedness kontrolü kelime kanıtı
kapısıyla aynı sebepten çöküyor demektir ve cross-encoder'a geçilmelidir.

--- Ön kapı ölçümü ---

Aynı çalıştırma, gevşetilecek kelime kanıtı kapısı için de veri üretir: bütün
alakalı vakaların ağırlıklı kapsama minimumu ile kapsam dışı ("hava nasıl",
"çikolatalı kek") vakaların maksimumu. Ön kapı artık cevabın varlığına değil
yalnızca **sorunun bu korpusun konusu olup olmadığına** karar verecek; eşiği bu
iki sayının arasına konur.
"""

import json
import random
from pathlib import Path

from app.database import get_all_chunks
from app.groundedness import (
    groundedness_score,
    sentence_support,
    split_sentences,
)
from app.term_evidence import build_context_terms
from app.rag_service import RAGService
from app.retrieval import get_top_chunks
from app.term_evidence import term_coverage

CASES_PATH = Path("eval_cases.json")
UNRELATED_SENTENCES_PER_CASE = 3


def load_cases():
    with CASES_PATH.open(encoding="utf-8") as handle:
        return json.load(handle)


def context_for(question, service):
    """Uygulamanın modele gerçekten verdiği chunk listesini üretir.

    Araç kendi context seçim kopyasını tutmaz; tutsaydı ölçülen şey uygulamanın
    çalıştırdığı şey olmazdı. Aynı gerekçe `tools/chunking_analysis.py` içinde
    `rank_chunks()`in ayrılmasının da sebebidir.
    """
    results = get_top_chunks(question)

    if not results:
        return [], None

    matched = service.select_matched_context_chunks(results, question=question)
    context = service.order_context_chunks(
        service.expand_context_chunks(matched),
        matched,
    )

    return context, results[0].get("question_term_weights")


def unrelated_sentences(chunks, exclude_source, rng):
    """Context'e girmeyen BAŞKA bir dokümandan cümleler."""
    pool = [
        chunk
        for chunk in chunks
        if chunk["source_name"] != exclude_source
    ]

    if not pool:
        return []

    sentences = []

    for chunk in rng.sample(pool, min(len(pool), UNRELATED_SENTENCES_PER_CASE)):
        for sentence in split_sentences(chunk["chunk_text"]):
            if len(sentence.split()) >= 5:
                sentences.append(sentence)
                break

    return sentences


def summarize(label, values):
    if not values:
        print(f"{label:<12} ölçüm yok")
        return None, None

    low = min(values)
    high = max(values)
    average = sum(values) / len(values)
    print(
        f"{label:<12} n={len(values):<4} min={low:.4f}  "
        f"ort={average:.4f}  max={high:.4f}"
    )

    return low, high


def main():
    rng = random.Random(20260731)
    cases = load_cases()
    service = RAGService()
    all_chunks = get_all_chunks()

    grounded = []
    paraphrase = []
    ungrounded = []
    relevant_gate = []
    out_of_scope_gate = []
    hard_negative_gate = []

    paraphrase_worst = []
    ungrounded_worst = []
    misretrieved = []
    fabricated_answers = []
    grounded_answers = []

    for case in cases:
        question = case["question"]
        context, weights = context_for(question, service)

        if not context:
            continue

        coverage = term_coverage(question, context, weights=weights)

        if case["expectation"] == "relevant":
            if coverage is not None:
                relevant_gate.append((coverage, case["name"]))

            # Retrieval yanlış dokümanı getirdiyse groundedness'ın düşük çıkması
            # DOĞRU davranıştır, kalibrasyon verisi değil. Bu vakaları eşiğe
            # dahil etmek eşiği yapay olarak sıfıra çeker ve ölçümü bozar; bunlar
            # sıralama sorunudur ve reranking'in konusudur.
            if context[0]["source_name"] != case.get("expected_source"):
                misretrieved.append(case["name"])
                continue

            context_terms = build_context_terms(context)

            for chunk in context:
                for sentence in split_sentences(chunk["chunk_text"]):
                    support = sentence_support(sentence, context_terms)
                    if support is not None:
                        grounded.append(support)

            for sentence in split_sentences(question):
                support = sentence_support(sentence, context_terms)
                if support is not None:
                    paraphrase.append(support)
                    paraphrase_worst.append((support, case["name"]))

            fabricated = unrelated_sentences(
                all_chunks,
                context[0]["source_name"],
                rng,
            )

            for sentence in fabricated:
                support = sentence_support(sentence, context_terms)
                if support is not None:
                    ungrounded.append(support)
                    ungrounded_worst.append((support, case["name"], sentence))

            # Cevap seviyesi vekil: çok cümleli uydurma bir cevap. Tek cümlelik
            # metinde oran yalnızca 0.00 veya 1.00 olabildiği için cevap eşiği
            # ancak böyle ölçülebilir.
            if len(fabricated) >= 2:
                score = groundedness_score(" ".join(fabricated), context)
                if score is not None:
                    fabricated_answers.append(score)

            score = groundedness_score(
                " ".join(
                    sentence
                    for chunk in context
                    for sentence in split_sentences(chunk["chunk_text"])[:1]
                ),
                context,
            )
            if score is not None:
                grounded_answers.append(score)
        elif coverage is not None:
            if case["name"].startswith("hard_negative"):
                hard_negative_gate.append((coverage, case["name"]))
            else:
                out_of_scope_gate.append((coverage, case["name"]))

    print("=== CÜMLE SEVİYESİ (GROUNDEDNESS_SENTENCE_SUPPORT) ===")
    summarize("DAYANAKLI", grounded)
    summarize("PARAFRAZ", paraphrase)
    summarize("DAYANAKSIZ", ungrounded)

    print("\nEşik adaylarında cümle bazlı hata oranları:")
    print("  eşik   parafraz geçen   dayanaksız geçen")
    for candidate in (0.34, 0.40, 0.50, 0.60, 0.67, 0.75, 1.00):
        legit = sum(1 for value in paraphrase if value >= candidate)
        false_pass = sum(1 for value in ungrounded if value >= candidate)
        print(
            f"  {candidate:.2f}   "
            f"{legit:>4}/{len(paraphrase):<4} ({legit / len(paraphrase):.1%})   "
            f"{false_pass:>4}/{len(ungrounded):<4} "
            f"({false_pass / len(ungrounded):.1%})"
        )

    print("\n=== CEVAP SEVİYESİ (GROUNDEDNESS_THRESHOLD) ===")
    grounded_answer_min, _ = summarize("DAYANAKLI", grounded_answers)
    _, fabricated_max = summarize("UYDURMA", fabricated_answers)

    if grounded_answer_min is not None and fabricated_max is not None:
        gap = grounded_answer_min - fabricated_max
        print(f"\nBOŞLUK: {gap:+.4f}")
        if gap > 0:
            print(
                f"Güvenli aralık: {fabricated_max:.4f} - {grounded_answer_min:.4f}"
            )
            print(
                f"Aralığın ortası: "
                f"{(fabricated_max + grounded_answer_min) / 2:.4f}"
            )
        else:
            print("Ayrım yok; cross-encoder gerekir.")

    print(f"\nSıralama hatası yüzünden dışlanan vaka: {len(misretrieved)}")
    for name in misretrieved:
        print(f"  {name}")

    print("\nEn düşük parafraz skorları:")
    for score, name in sorted(paraphrase_worst)[:8]:
        print(f"  {score:.4f}  {name}")

    print("\nEn yüksek dayanaksız skorlar (yanlış geçen cümleler):")
    for score, name, sentence in sorted(ungrounded_worst, reverse=True)[:8]:
        print(f"  {score:.4f}  [{name}] {sentence[:90]}")

    above = [value for value in ungrounded if value >= 0.50]
    print(
        f"\nDayanaksız cümlelerin {len(above)}/{len(ungrounded)} tanesi "
        f"0.50 ve üstünde."
    )

    print("\n=== ÖN KAPI (soru -> context, ağırlıklı) ===")
    print("Alakalı vakaların en düşükleri:")
    for coverage, name in sorted(relevant_gate)[:8]:
        print(f"  {coverage:.4f}  {name}")

    print("\nKapsam dışı vakalar:")
    for coverage, name in sorted(out_of_scope_gate, reverse=True):
        print(f"  {coverage:.4f}  {name}")

    print("\nHard negative vakalar (artık LLM'e gidecekler):")
    for coverage, name in sorted(hard_negative_gate, reverse=True):
        print(f"  {coverage:.4f}  {name}")

    if relevant_gate and out_of_scope_gate:
        relevant_min = min(coverage for coverage, _ in relevant_gate)
        out_max = max(coverage for coverage, _ in out_of_scope_gate)
        print(f"\nÖn kapı boşluğu: {relevant_min - out_max:+.4f}")
        if relevant_min > out_max:
            print(f"Güvenli aralık: {out_max:.4f} - {relevant_min:.4f}")
            print(f"Aralığın ortası: {(out_max + relevant_min) / 2:.4f}")


if __name__ == "__main__":
    main()
