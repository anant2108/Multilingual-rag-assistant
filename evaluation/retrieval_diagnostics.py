
from pathlib import Path

from config import VECTOR_STORE_DIR
from models.embedding import get_embedding_model
from vector_store.faiss_db import load_vector_store


QUERIES = {
    "PARAPHRASED": "Explain the stakeholder map.",
    "EXACT TITLE": "Stakeholder map: concentric-circle model",
    "KEY CONCEPTS": (
        "Core internal control; middle operational influence; "
        "outer macro governance pressure."
    ),
    "BGE QUERY INSTRUCTION": (
        "Represent this sentence for searching relevant passages: "
        "Explain the stakeholder map."
    ),
}



def main():
    embedding_model = get_embedding_model()
    vector_store = load_vector_store(
        embedding_model,
        VECTOR_STORE_DIR,
    )

    for category, query in QUERIES.items():
        print(f"\n{'=' * 60}")
        print(f"{category}: {query}")
        print("=" * 60)

        results = vector_store.similarity_search_with_score(
            query,
            k=4,
        )

        for rank, (document, distance) in enumerate(results, start=1):
            metadata = document.metadata or {}
            source = Path(
                str(metadata.get("source", "Unknown source"))
            ).name

            page = metadata.get("page")
            page_label = (
                page + 1 if isinstance(page, int) else "Unknown"
            )

            snippet = " ".join(document.page_content.split())[:220]

            print(
                f"\nRank: {rank} | "
                f"Distance: {float(distance):.4f}"
            )
            print(f"Source: {source} | Page: {page_label}")
            print(f"Text: {snippet}")


if __name__ == "__main__":
    main()
