
from pathlib import Path

from config import VECTOR_STORE_DIR
from models.embedding import get_embedding_model
from vector_store.faiss_db import load_vector_store


# Page numbers below are one-based, as displayed by our application.
TEST_CASES = [
    {
        "name": "Internal stakeholders",
        "query": "What are internal stakeholders?",
        "answerable": True,
        "expected_pages": {3, 4},
    },
    {
        "name": "Stakeholder definition",
        "query": "What is a stakeholder?",
        "answerable": True,
        "expected_pages": {2},
    },
    {
        "name": "External stakeholders",
        "query": "Who are external stakeholders?",
        "answerable": True,
        "expected_pages": {3},
    },
    {
        "name": "Stakeholder map",
        "query": "Explain the stakeholder map.",
        "answerable": True,
        "expected_pages": {4},
    },
    {
        "name": "Japan capital",
        "query": "What is the capital of Japan?",
        "answerable": False,
        "expected_pages": set(),
    },
    {
        "name": "Photosynthesis",
        "query": "Explain photosynthesis in plants.",
        "answerable": False,
        "expected_pages": set(),
    },
    {
        "name": "Python programming",
        "query": "How do I sort a Python list?",
        "answerable": False,
        "expected_pages": set(),
    },
    {
        "name": "Currency exchange",
        "query": "What is today's USD to INR exchange rate?",
        "answerable": False,
        "expected_pages": set(),
    },
]


def main():
    embedding_model = get_embedding_model()

    vector_store = load_vector_store(
        embedding_model,
        VECTOR_STORE_DIR,
    )

    k = 4
    answerable_tests = 0
    retrieval_hits = 0

    print(f"\nRunning {len(TEST_CASES)} retrieval tests")
    print(f"Retrieval depth: top {k}\n")

    for case in TEST_CASES:
        results = vector_store.similarity_search_with_score(
            case["query"],
            k=k,
        )

        retrieved_pages = set()
        best_distance = float("inf")

        for document, distance in results:
            best_distance = min(
                best_distance,
                float(distance),
            )

            page = (document.metadata or {}).get("page")

            if isinstance(page, int):
                retrieved_pages.add(page + 1)

        if case["answerable"]:
            answerable_tests += 1

            hit = bool(
                retrieved_pages & case["expected_pages"]
            )

            if hit:
                retrieval_hits += 1

            status = "HIT" if hit else "MISS"
        else:
            # We have no calibrated abstention threshold yet.
            status = "UNANSWERABLE TEST"

        print(f"[{status}] {case['name']}")
        print(f"  Query: {case['query']}")
        print(f"  Best distance: {best_distance:.4f}")
        print(f"  Retrieved pages: {sorted(retrieved_pages)}")

    print("\n===== SUMMARY =====")
    print(f"Answerable tests: {answerable_tests}")
    print(f"Relevant-page hits: {retrieval_hits}")

    if answerable_tests:
        hit_rate = retrieval_hits / answerable_tests
        print(f"Hit@{k}: {hit_rate:.1%}")

    print(
        "\nNote: unanswerable cases are diagnostic examples, "
        "not yet automatic pass/fail tests. "
        "We must calibrate the evidence gate separately."
    )


if __name__ == "__main__":
    main()
