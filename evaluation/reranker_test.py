
import torch
from sentence_transformers import CrossEncoder
from langchain_community.retrievers import BM25Retriever

from config import DATA_DIR, VECTOR_STORE_DIR
from loaders.loader import load_documents
from processing.chunker import split_documents
from models.embedding import get_embedding_model
from vector_store.faiss_db import load_vector_store


RETRIEVAL_K = 8
RERANK_TOP_K = 4
RERANKER_MODEL = "BAAI/bge-reranker-base"

TEST_CASES = [
    {
        "name": "Internal stakeholders",
        "query": "What are internal stakeholders?",
        "answerable": True,
        "expected_pages": {3},
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


def document_key(document):
    """Generate a unique key for each retrieved chunk."""
    metadata = document.metadata or {}

    return (
        metadata.get("source"),
        metadata.get("page"),
        document.page_content,
    )


def get_page_number(document):
    """Convert zero-based PDF metadata into one-based page numbers."""
    page = (document.metadata or {}).get("page")

    if isinstance(page, int):
        return page + 1

    return None


def main():
    # Build a temporary BM25 keyword retriever.
    source_documents = load_documents(DATA_DIR)
    chunks = split_documents(source_documents)

    bm25 = BM25Retriever.from_documents(
        chunks,
        k=RETRIEVAL_K,
    )

    # Load our existing FAISS index.
    embedding_model = get_embedding_model()

    vector_store = load_vector_store(
        embedding_model,
        VECTOR_STORE_DIR,
    )

    # Load the reranker once, not once per question.
    mps = getattr(torch.backends, "mps", None)
    device = "mps" if mps and mps.is_available() else "cpu"

    print(f"Loading reranker on {device}...")

    reranker = CrossEncoder(
        RERANKER_MODEL,
        max_length=512,
        device=device,
    )

    answerable_total = 0
    dense_hits = 0
    candidate_hits = 0
    reranked_hits = 0
    reciprocal_rank_total = 0.0

    print(f"\nRunning {len(TEST_CASES)} evaluation questions")

    for case in TEST_CASES:
        query = case["query"]
        expected_pages = case["expected_pages"]

        # Stage 1: retrieve candidates using the original question.
        dense_results = vector_store.similarity_search(
            query,
            k=RETRIEVAL_K,
        )

        keyword_results = bm25.invoke(query)

        # Stage 2: combine and deduplicate candidates.
        candidates = {}

        for document in dense_results + keyword_results:
            candidates[document_key(document)] = document

        candidate_documents = list(candidates.values())

        # Stage 3: score each candidate against the original question.
        pairs = [
            (query, document.page_content)
            for document in candidate_documents
        ]

        scores = reranker.predict(
            pairs,
            batch_size=4,
            show_progress_bar=False,
        )

        ranked_results = sorted(
            zip(candidate_documents, scores),
            key=lambda item: float(item[1]),
            reverse=True,
        )

        # Inspect the relevant pages in each stage.
        dense_top_pages = {
            get_page_number(document)
            for document in dense_results[:RERANK_TOP_K]
        }

        candidate_pages = {
            get_page_number(document)
            for document in candidate_documents
        }

        reranked_top_pages = {
            get_page_number(document)
            for document, _ in ranked_results[:RERANK_TOP_K]
        }

        print("\n" + "=" * 60)
        print(f"QUESTION: {case['name']}")
        print(f"Query: {query}")

        if case["answerable"]:
            answerable_total += 1

            dense_hit = bool(
                dense_top_pages & expected_pages
            )

            candidate_hit = bool(
                candidate_pages & expected_pages
            )

            reranked_hit = bool(
                reranked_top_pages & expected_pages
            )

            if dense_hit:
                dense_hits += 1

            if candidate_hit:
                candidate_hits += 1

            if reranked_hit:
                reranked_hits += 1

            relevant_rank = next(
                (
                    rank
                    for rank, (document, _) in enumerate(
                        ranked_results,
                        start=1,
                    )
                    if get_page_number(document) in expected_pages
                ),
                None,
            )

            if relevant_rank is not None:
                reciprocal_rank_total += 1.0 / relevant_rank

            print(
                f"FAISS Hit@{RERANK_TOP_K}: "
                f"{'HIT' if dense_hit else 'MISS'}"
            )

            print(
                f"Candidate pool contains expected page: "
                f"{'YES' if candidate_hit else 'NO'}"
            )

            print(
                f"Reranked Hit@{RERANK_TOP_K}: "
                f"{'HIT' if reranked_hit else 'MISS'}"
            )

            print(f"Best relevant rank after reranking: {relevant_rank}")

        else:
            print("OUT-OF-DOCUMENT TEST")
            print("Top reranked passages (inspect manually):")

            for rank, (document, score) in enumerate(
                ranked_results[:3],
                start=1,
            ):
                print(
                    f"{rank}. Page {get_page_number(document)} "
                    f"| Score: {float(score):.4f}"
                )

        # Show the top four reranked passages.
        print("\nTop reranked pages:")

        for rank, (document, score) in enumerate(
            ranked_results[:RERANK_TOP_K],
            start=1,
        ):
            print(
                f"{rank}. Page {get_page_number(document)} "
                f"| Score: {float(score):.4f}"
            )

    print("\n" + "=" * 60)
    print("FINAL EVALUATION SUMMARY")

    if answerable_total:
        print(
            f"FAISS Hit@{RERANK_TOP_K}: "
            f"{dense_hits}/{answerable_total} "
            f"({dense_hits / answerable_total:.1%})"
        )

        print(
            f"Candidate pool coverage: "
            f"{candidate_hits}/{answerable_total} "
            f"({candidate_hits / answerable_total:.1%})"
        )

        print(
            f"Reranked Hit@{RERANK_TOP_K}: "
            f"{reranked_hits}/{answerable_total} "
            f"({reranked_hits / answerable_total:.1%})"
        )

        print(
            f"Reranked MRR: "
            f"{reciprocal_rank_total / answerable_total:.4f}"
        )

    print(
        "\nUnanswerable questions are diagnostic only. "
        "No confidence threshold has been calibrated yet."
    )


if __name__ == "__main__":
    main()
