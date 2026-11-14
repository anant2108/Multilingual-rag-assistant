import torch
import logging

from langchain_community.retrievers import BM25Retriever
from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)


RERANKER_MODEL_NAME = "BAAI/bge-reranker-base"


def build_keyword_retriever(chunks, candidate_k=8):
    """Build a BM25 keyword retriever from document chunks."""

    if not chunks:
        raise ValueError(
            "Cannot build keyword retriever without chunks."
        )

    return BM25Retriever.from_documents(
        chunks,
        k=candidate_k,
    )


def load_reranker(model_name=RERANKER_MODEL_NAME):
    """Load the reranker on the best available device."""

    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    return CrossEncoder(
        model_name,
        max_length=512,
        device=device,
    )


def document_key(document):
    """Generate a unique identifier for a document chunk."""

    metadata = document.metadata or {}

    return (
        metadata.get("source"),
        metadata.get("page"),
        document.page_content,
    )


def search(
    vector_store,
    keyword_retriever,
    reranker,
    query,
    candidate_k=8,
    final_k=4,
):
    """
    Retrieve candidates using dense and keyword search,
    then rerank them using a cross-encoder.
    """

    if not query or not query.strip():
        raise ValueError("Search query cannot be empty.")

    if candidate_k < 1 or final_k < 1:
        raise ValueError("Retrieval limits must be positive.")

    query = query.strip()

    # Stage 1: semantic retrieval using FAISS.
    dense_results = vector_store.similarity_search(
        query,
        k=candidate_k,
    )

    # Stage 2: keyword retrieval using BM25.
    keyword_results = keyword_retriever.invoke(query)

    # Stage 3: combine and deduplicate candidates.
    candidates = {}

    for document in dense_results + keyword_results:
        key = document_key(document)
        candidates.setdefault(key, document)

    candidate_documents = list(candidates.values())

    if not candidate_documents:
        return []

    # Stage 4: score each candidate against the original query.
    pairs = [
        (query, document.page_content)
        for document in candidate_documents
    ]

    scores = reranker.predict(
        pairs,
        batch_size=4,
        show_progress_bar=False,
    )

    # Stage 5: sort candidates by reranker relevance score.
    ranked_results = sorted(
        zip(candidate_documents, scores),
        key=lambda item: float(item[1]),
        reverse=True,
    )

    logger.info(
        "Retrieved %d unique candidates; returning top %d",
        len(candidate_documents),
        min(final_k, len(ranked_results)),
    )

    return [
        document
        for document, score in ranked_results[:final_k]
    ]
