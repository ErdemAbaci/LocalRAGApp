"""Kelime kanıtı sinyalinin ayırt ediciliğini ölçer. (Keşif aracı)

Bu dosya uygulamanın parçası DEĞİLDİR. `main.py` ve `app/` bunu içe aktarmaz;
eval de çalıştırmaz. Amacı bir tasarım kararını veriyle beslemektir ve tek
başına çalıştırılır.

Cevaplamaya çalıştığı soru: "sorunun kelimeleri modele giden metinde gerçekten
geçiyor mu" sinyali, cevabı dokümanda bulunan soruları bulunmayanlardan
ayırabiliyor mu? Ve Türkçe ekleri ele almanın hangi yöntemi bu ayrımı en iyi
korur?

Çalıştırma (repository kökünden):

    source .venv/bin/activate
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python tools/term_evidence_analysis.py

Ölçülen eşleştiriciler:

- `substring` : soru kelimesi metnin herhangi bir yerinde geçiyor mu.
- `whole`     : soru kelimesi metinde tam bir kelime olarak geçiyor mu.
- `prefixN`   : soru kelimesi ile bir metin kelimesi, en az N karakterlik ortak
                önek paylaşıp paylaşmadığına bakar ve kısa olanın uzun olanın
                öneki olmasını şart koşar. Türkçe eklemeli olduğu için kök
                baştadır; bu yüzden önek eşleştirmesi dilbilgisel olarak
                anlamlıdır.

Kritik ayrım: kelimeyi N karaktere kesip metnin içinde aramak (eski deneme)
`sayısı` -> `sayısal` gibi yanlış eşleşmeler üretir. Tam kelime öneki şartı bunu
üretmez, çünkü `sayısı` kelimesi `sayısal` kelimesinin öneki değildir.

Çıktı, her eşleştirici için alakalı ve hard negative gruplarının aralığını ve
aradaki boşluğu raporlar. Boşluk ne kadar büyükse eşik seçimi o kadar güvenlidir.

Ayrıntı ve sonuçların yorumu için AGENTS.md bölüm 7.
"""

import json
from pathlib import Path

from app.config import TOP_K
from app.database import get_all_chunks
from app.rag_service import RAGService
from app.sparse_search import inverse_document_frequency
from app.retrieval import gate_score, get_top_chunks
from app.term_evidence import (
    common_prefix_length,
    extract_question_terms,
    normalize_text,
    tokenize,
)


def content_words(question):
    """Uygulamanın gerçek terim çıkarımını kullanır.

    Araç kendi stopword kopyasını tutmaz. Tutsaydı ölçüm, uygulamanın gerçekte
    kullandığı listeden sapabilir ve eşik kararı yanlış veriye dayanırdı.
    """
    return extract_question_terms(question)


def match_substring(word, context_text, context_words):
    return word in context_text


def match_whole_word(word, context_text, context_words):
    return word in context_words


def make_prefix_matcher(min_prefix):
    def match_prefix(word, context_text, context_words):
        if word in context_words:
            return True

        for context_word in context_words:
            shorter, longer = sorted((word, context_word), key=len)

            if len(shorter) < min_prefix:
                continue

            if longer.startswith(shorter):
                return True

        return False

    return match_prefix


def make_common_prefix_matcher(min_prefix):
    """Kelimelerden hiçbiri diğerinin öneki olmasa da ortak kökü arar.

    `korunulur` ve `korunmak` aynı kökten türer ama hiçbiri diğerinin öneki
    değildir; "kısa olan uzun olanın önekidir" kuralı bu meşru eşleşmeyi
    kaçırır. Ortak önek kuralı yakalar.

    Tam eşleşme her zaman önce denenir; aksi halde `rag` gibi minimum kökten
    kısa kelimeler hiç eşleşemez.
    """
    def match_common_prefix(word, context_text, context_words):
        if word in context_words:
            return True

        return any(
            common_prefix_length(word, context_word) >= min_prefix
            for context_word in context_words
        )

    return match_common_prefix


def make_root_matcher(min_prefix, min_short):
    """Ortak kök kuralına "kısa kelime tamamen tükendi" istisnasını ekler.

    Ölçüm bir yanlış eksiklik gösterdi: `avından` kelimesi korpusta hiçbir şeyle
    eşleşmiyordu, çünkü metindeki karşılığı `avı` yalnızca 3 karakter ve minimum
    ortak kök 5. Oysa `avı`, `avından` kelimesinin tamamen tükenen bir önekidir;
    bu Türkçe'de kökün tam olarak eşleştiği durumdur.

    Kısa kelimeler için bu istisna gevşetme değil, kuralın asıl halidir: ortak
    kök şartı uzun kelimelerde yanlış eşleşmeyi engellemek için var, kökün
    kendisi kısa olduğunda ise şart kökü tamamen kapsamak olmalıdır.
    """
    def match_root(word, context_text, context_words):
        if word in context_words:
            return True

        for context_word in context_words:
            length = common_prefix_length(word, context_word)

            if length >= min_prefix:
                return True

            shorter = min(len(word), len(context_word))

            if shorter >= min_short and length == shorter:
                return True

        return False

    return match_root


MATCHERS = {
    "prefix5": make_prefix_matcher(5),
    "common4": make_common_prefix_matcher(4),
    "common5": make_common_prefix_matcher(5),
    "common6": make_common_prefix_matcher(6),
    "common7": make_common_prefix_matcher(7),
    "kök5-3": make_root_matcher(5, 3),
    "kök5-4": make_root_matcher(5, 4),
}

# Uygulamanın kullandığı eşleştirici.
APPLIED_MATCHER = "kök5-3"


def build_context(results):
    service = RAGService()
    matched = service.select_matched_context_chunks(results)

    if not matched:
        return "", set()

    context_chunks = service.order_context_chunks(
        service.expand_context_chunks(matched),
        matched,
    )
    text = normalize_text("\n".join(
        chunk["chunk_text"] for chunk in context_chunks
    ))

    return text, set(tokenize(text))


def coverage(words, matcher, context_text, context_words):
    if not words:
        return None, []

    matched = [
        word
        for word in words
        if matcher(word, context_text, context_words)
    ]

    return len(matched) / len(words), matched


def classify(case):
    if case["expectation"] == "relevant":
        return "ALAKALI"

    if case.get("difficulty") == "hard":
        return "TUZAK"

    return "kolay-neg"


def corpus_documents():
    """Korpustaki her chunk için token listesi. IDF'in denklemi budur."""
    return [
        (normalize_text(chunk["chunk_text"]), set(tokenize(chunk["chunk_text"])))
        for chunk in get_all_chunks()
    ]


def matcher_term_weights(words, matcher, documents):
    """Bir eşleştiriciye göre kelime -> IDF ağırlığı.

    Ağırlık eşleştiriciden bağımsız değildir: `avından` kelimesinin korpusta
    kaç dokümanda geçtiği, `avı` ile eşleşip eşleşmediğine bağlıdır. Bu yüzden
    her eşleştirici kendi ağırlıklarıyla ölçülür; retrieval'ın hesapladığı
    ağırlıkları kullanmak yalnızca uygulamadaki eşleştirici için doğru olurdu.

    IDF formülü `app/sparse_search` içinden alınır; araç kendi kopyasını tutmaz.
    """
    weights = {}

    for word in words:
        matching = sum(
            1
            for document_text, document_words in documents
            if matcher(word, document_text, document_words)
        )
        weights[word] = inverse_document_frequency(len(documents), matching)

    return weights


def weighted_coverage(words, matched, weights):
    if not words:
        return None

    total = sum(weights.get(word, 1.0) for word in words)

    if total <= 0:
        return None

    return sum(weights.get(word, 1.0) for word in matched) / total


def collect_rows():
    cases = json.loads(Path("eval_cases.json").read_text(encoding="utf-8"))
    documents = corpus_documents()
    rows = []

    for case in cases:
        results = get_top_chunks(case["question"], top_k=TOP_K)
        context_text, context_words = build_context(results)
        words = content_words(case["question"])

        row = {
            "group": classify(case),
            "name": case["name"],
            "score": gate_score(results),
            "words": words,
            "ratios": {},
            "weighted": {},
            "matched": {},
            "weights": {},
        }

        for matcher_name, matcher in MATCHERS.items():
            ratio, matched = coverage(words, matcher, context_text, context_words)
            weights = matcher_term_weights(words, matcher, documents)

            row["ratios"][matcher_name] = ratio
            row["matched"][matcher_name] = matched
            row["weights"][matcher_name] = weights
            row["weighted"][matcher_name] = weighted_coverage(words, matched, weights)

        rows.append(row)

    return rows


def print_case_table(rows):
    header = f"{'GRUP':<10} {'VAKA':<34}{'SKOR':>7}"
    for name in MATCHERS:
        header += f"{name:>8}"
    print("ORAN (kelime sayısına göre kapsama)\n")
    print(header)
    print("-" * len(header))

    for row in sorted(rows, key=lambda item: (item["group"], -item["score"])):
        line = f"{row['group']:<10} {row['name']:<34}{row['score']:>7.4f}"
        for name in MATCHERS:
            ratio = row["ratios"][name]
            line += f"{'-' if ratio is None else f'{ratio:.2f}':>8}"
        print(line)

    print("\n\nAĞIRLIKLI (IDF ile ayırt ediciliğe göre kapsama)\n")
    print(header)
    print("-" * len(header))

    for row in sorted(rows, key=lambda item: (item["group"], -item["score"])):
        line = f"{row['group']:<10} {row['name']:<34}{row['score']:>7.4f}"
        for name in MATCHERS:
            ratio = row["weighted"][name]
            line += f"{'-' if ratio is None else f'{ratio:.2f}':>8}"
        print(line)


def group_range(rows, key, name):
    relevant = [
        row[key][name]
        for row in rows
        if row["group"] == "ALAKALI" and row[key][name] is not None
    ]
    traps = [
        row[key][name]
        for row in rows
        if row["group"] == "TUZAK" and row[key][name] is not None
    ]

    if not relevant or not traps:
        return None

    return min(relevant), max(traps)


def print_separation(rows):
    print("\n\nAYRIM GÜCÜ (alakalı en düşük  vs  tuzak en yüksek)\n")
    header = (
        f"{'EŞLEŞTİRİCİ':<10}"
        f"{'oran alk':>10}{'oran tzk':>10}{'BOŞLUK':>9}"
        f"{'ağr alk':>10}{'ağr tzk':>10}{'BOŞLUK':>9}"
    )
    print(header)
    print("-" * len(header))

    best = None

    for name in MATCHERS:
        plain = group_range(rows, "ratios", name)
        weighted = group_range(rows, "weighted", name)

        if plain is None or weighted is None:
            continue

        plain_gap = plain[0] - plain[1]
        weighted_gap = weighted[0] - weighted[1]

        print(
            f"{name:<10}{plain[0]:>10.2f}{plain[1]:>10.2f}{plain_gap:>9.2f}"
            f"{weighted[0]:>10.2f}{weighted[1]:>10.2f}{weighted_gap:>9.2f}"
        )

        for label, gap, bounds in (
            ("oran", plain_gap, plain),
            ("ağırlıklı", weighted_gap, weighted),
        ):
            if gap > 0 and (best is None or gap > best[1]):
                best = (f"{name} / {label}", gap, bounds)

    if best is None:
        print("\nHiçbir ayar iki grubu ayırmıyor.")
        return

    label, gap, bounds = best
    print(
        f"\nEn geniş boşluk: {label} ({gap:.2f}). "
        f"Güvenli eşik aralığı {bounds[1]:.2f} ile {bounds[0]:.2f} arasıdır."
    )


def print_weight_detail(rows, matcher_name):
    print(f"\n\nAĞIRLIK DETAYI ({matcher_name}; kelime: idf, * = eşleşti)\n")

    for row in sorted(rows, key=lambda item: (item["group"], -item["score"])):
        matched = set(row["matched"][matcher_name])
        weights = row["weights"][matcher_name]
        parts = [
            f"{word}: {weights.get(word, 1.0):.2f}{'*' if word in matched else ''}"
            for word in row["words"]
        ]
        ratio = row["ratios"][matcher_name]
        weighted = row["weighted"][matcher_name]
        print(f"[{row['group']}] {row['name']}")
        print(
            f"    oran={'-' if ratio is None else f'{ratio:.2f}'}  "
            f"ağırlıklı={'-' if weighted is None else f'{weighted:.2f}'}"
        )
        print(f"    {', '.join(parts) or '(içerik kelimesi yok)'}")
        print()


def main():
    rows = collect_rows()
    print_case_table(rows)
    print_separation(rows)
    print_weight_detail(rows, APPLIED_MATCHER)


if __name__ == "__main__":
    main()
