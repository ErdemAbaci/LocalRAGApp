import json
import math
from pathlib import Path

from app.config import SIMILARITY_THRESHOLD, TOP_K
from app.database import get_all_chunks, get_chunk_stats
from app.retrieval import get_top_chunks


EVAL_CASES_PATH = Path(__file__).with_name("eval_cases.json")
EXPECTED_EMBEDDING_DIMENSION = 384


def load_eval_cases():
    return json.loads(EVAL_CASES_PATH.read_text(encoding="utf-8"))


def validate_index():
    chunks = get_all_chunks()

    if not chunks:
        return False, "Index boş. Önce /reindex çalıştır."

    for chunk in chunks:
        embedding = chunk["embedding"]

        if len(embedding) != EXPECTED_EMBEDDING_DIMENSION:
            return False, (
                f"chunk_id={chunk['id']} embedding boyutu {len(embedding)}; "
                f"beklenen {EXPECTED_EMBEDDING_DIMENSION}."
            )

        if not all(math.isfinite(value) for value in embedding):
            return False, f"chunk_id={chunk['id']} geçersiz embedding değeri içeriyor."

    return True, f"{len(chunks)} chunk ve embedding değerleri sağlıklı."


def evaluate_relevant_case(case, results):
    if not results:
        return False, "Retrieval sonucu gelmedi."

    best_result = results[0]
    expected_source = case["expected_source"]
    min_score = case.get("min_score", SIMILARITY_THRESHOLD)

    if best_result["source_name"] != expected_source:
        return False, (
            f"beklenen kaynak={expected_source}, "
            f"gelen={best_result['source_name']}"
        )

    if best_result["score"] < min_score:
        return False, (
            f"skor={best_result['score']:.4f}, "
            f"minimum={min_score:.4f}"
        )

    return True, (
        f"kaynak={best_result['source_name']}, "
        f"skor={best_result['score']:.4f}"
    )


def evaluate_not_found_case(case, results):
    if not results:
        return True, "Sonuç yok; beklenen davranış."

    best_score = results[0]["score"]
    max_score = case.get("max_score", SIMILARITY_THRESHOLD)

    if best_score >= max_score:
        return False, f"skor={best_score:.4f}, maksimum={max_score:.4f}"

    return True, f"skor={best_score:.4f}, eşik altında"


def evaluate_case(case):
    results = get_top_chunks(case["question"], top_k=TOP_K)
    expectation = case["expectation"]

    if expectation == "relevant":
        return evaluate_relevant_case(case, results)

    if expectation == "not_found":
        return evaluate_not_found_case(case, results)

    return False, f"Bilinmeyen expectation: {expectation}"


def main():
    stats = get_chunk_stats()

    if stats["total_chunks"] == 0:
        print("Index boş. Önce uygulamada /reindex çalıştır.")
        return 1

    index_passed, index_detail = validate_index()
    index_status = "PASS" if index_passed else "FAIL"
    print(f"{index_status:<5} index_health - {index_detail}")

    passed_count = 1 if index_passed else 0
    cases = load_eval_cases()

    for case in cases:
        passed, detail = evaluate_case(case)
        status = "PASS" if passed else "FAIL"
        print(f"{status:<5} {case['name']} - {detail}")

        if passed:
            passed_count += 1

    total_count = len(cases) + 1
    print(f"\n{passed_count}/{total_count} test başarılı")

    return 0 if passed_count == total_count else 1


if __name__ == "__main__":
    raise SystemExit(main())
