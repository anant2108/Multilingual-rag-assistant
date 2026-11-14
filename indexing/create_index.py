import logging

from config import DATA_DIR, VECTOR_STORE_DIR, BATCH_SIZE
from loaders.loader import load_documents
from processing.chunker import split_documents
from models.embedding import get_embedding_model
from vector_store.faiss_db import (
    create_vector_store,
    save_vector_store,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
)

logger = logging.getLogger(__name__)


def main():
    # 1. Load source documents
    documents = load_documents(DATA_DIR)
    logger.info("Loaded %d document pages/records", len(documents))

    # 2. Split documents while retaining source metadata
    chunks = split_documents(documents)
    logger.info("Created %d chunks", len(chunks))

    # 3. Initialize the embedding model
    embedding_model = get_embedding_model()
    logger.info("Embedding model initialized")

    # 4. Create the searchable FAISS index
    vector_store = create_vector_store(
        chunks,
        embedding_model,
        batch_size=BATCH_SIZE,
    )

    # 5. Persist the index for later queries
    save_vector_store(vector_store, VECTOR_STORE_DIR)

    logger.info("Indexing completed successfully")


if __name__ == "__main__":
    main()