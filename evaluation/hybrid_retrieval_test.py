
import re

from langchain_community.retrievers import BM25Retriever

from config import DATA_DIR, VECTOR_STORE_DIR
from loaders.loader import load_documents
from processing.chunker import split_documents
from models.embedding import get_embedding_model
from vector_store.faiss_db import load_vector_store


QUESTION = "Explain the stakeholder map."

RETRIEVAL_K = 8
RRF_CONSTANT = 60

# PyPDFLoader stores page numbers starting from zero.
# Page 4 in the PDF therefore has metadata page=3.
TARGET_PAGE = 3


def normalize_query(query: str) -> str:
    """Remove common question prefixes to create a focused query."""
    query = query.strip().rstrip("?.! ")

    query = re.sub(
        r"^(?:please\s+)?(?:can you\s+)?"
        r"(?:explain|describe|summarize|"
        r"tell me about|what is|what are|who is|who are)\s+",
        "",
        query,
        flags=re.IGNORECASE,
    )

    query = re.sub(
        r"^(?:the|a|an)\s+",
        "",
        query,
        flags=re.IGNORECASE,
    )

    return query.strip()


def document_key(document):
    """Identify the same chunk across retrieval methods."""
    metadata = document.metadata or {}

    return (
        metadata.get("source"),
        metadata.get("page"),
        document.page_content,
    )


def reciprocal_rank_fusion(result_lists):
    """Combine ranked lists using Reciprocal Rank Fusion."""
    scores = {}
    documents_by_key = {}

    for results in result_lists:
        seen_in_list = set()

        for rank, document in enumerate(results, start=1):
            key = document_key(document)

            # Count a document at most once per ranked list.
            if key in seen_in_list:
                continue

            seen_in_list.add(key)
            documents_by_key[key] = document

            scores[key] = scores.get(key, 0.0) + (
                1.0 / (RRF_CONSTANT + rank)
            )

    ranked_keys = sorted(
        scores,
        key=scores.get,
        reverse=True,
    )

    return [
        (documents_by_key[key], scores[key])
        for key in ranked_keys
    ]


def show_documents(label, documents):
    """Print retrieved documents without misleading score values."""
    print(f"\n{label}")

    for rank, document in enumerate(documents, start=1):
        metadata = document.metadata or {}
        page = metadata.get("page")

        page_label = (
            page + 1 if isinstance(page, int) else "Unknown"
        )

        snippet = " ".join(document.page_content.split())[:150]

        print(f"{rank}. Page {page_label}: {snippet}")


def show_fused_results(results):
    """Print fused rankings and their RRF scores."""
    print("\n===== RRF FUSED RESULTS =====")

    for rank, (document, score) in enumerate(results, start=1):
        metadata = document.metadata or {}
        page = metadata.get("page")

        page_label = (
            page + 1 if isinstance(page, int) else "Unknown"
        )

        snippet = " ".join(document.page_content.split())[:150]

        print(
            f"{rank}. Page {page_label} | "
            f"RRF score: {score:.4f} | {snippet}"
        )


def main():
    # Build a keyword retriever from the same source documents.
    source_documents = load_documents(DATA_DIR)
    chunks = split_documents(source_documents)

    bm25 = BM25Retriever.from_documents(
        chunks,
        k=RETRIEVAL_K,
    )

    # Load the existing FAISS index; do not rebuild it.
    embedding_model = get_embedding_model()

    vector_store = load_vector_store(
        embedding_model,
        VECTOR_STORE_DIR,
    )

    normalized_query = normalize_query(QUESTION)

    print("Original query:  ", QUESTION)
    print("Normalized query:", normalized_query)

    result_lists = []

    # Compare the original question with its normalized form.
    queries = list(dict.fromkeys([
    QUESTION,
    normalized_query,
    "Stakeholder map: concentric-circle model",
    (
        "Core internal control, middle operational influence, "
        "outer macro governance pressure"
    ),
]))

    for query in queries:
        print(f"\n{'=' * 60}")
        print(f"QUERY: {query}")

        dense_results = vector_store.similarity_search(
            query,
            k=RETRIEVAL_K,
        )

        keyword_results = bm25.invoke(query)

        result_lists.extend(
            [dense_results, keyword_results]
        )

        show_documents(
            "FAISS semantic search",
            dense_results,
        )

        show_documents(
            "BM25 keyword search",
            keyword_results,
        )

    # Fuse all ranked lists.
    fused_results = reciprocal_rank_fusion(result_lists)

    # Find the known relevant PDF page in the fused ranking.
    target_page_results = [
        (rank, document, score)
        for rank, (document, score) in enumerate(
            fused_results,
            start=1,
        )
        if (document.metadata or {}).get("page") == TARGET_PAGE
    ]

    print("\n===== PAGE 4 RANK CHECK =====")

    if target_page_results:
        for rank, document, score in target_page_results:
            print(f"Page 4 found at fused rank {rank}")
            print(f"RRF score: {score:.4f}")
            print(document.page_content[:300])
    else:
        print("Page 4 was not found in the fused results.")

    show_fused_results(fused_results)


if __name__ == "__main__":
    main()
