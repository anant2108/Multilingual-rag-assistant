import logging
from pathlib import Path

from langchain_community.vectorstores import FAISS

from config import BATCH_SIZE, VECTOR_STORE_DIR

logger = logging.getLogger(__name__)


def create_vector_store(chunks, embedding_model, batch_size=BATCH_SIZE):
    """Embed chunks in batches and build a FAISS vector store."""
    if not chunks:
        raise ValueError("Cannot build an index without chunks.")

    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero.")

    first_batch = chunks[:batch_size]

    vector_store = FAISS.from_documents(
        first_batch,
        embedding_model,
    )

    logger.info(
        "Processed %d / %d chunks",
        len(first_batch),
        len(chunks),
    )

    for start in range(batch_size, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]

        vector_store.add_documents(batch)

        processed = min(start + len(batch), len(chunks))

        logger.info(
            "Processed %d / %d chunks",
            processed,
            len(chunks),
        )

    return vector_store


def save_vector_store(vector_store, save_path=VECTOR_STORE_DIR):
    """Save the vector store to disk."""
    path = Path(save_path)
    path.mkdir(parents=True, exist_ok=True)

    vector_store.save_local(str(path))

    logger.info("Saved FAISS index to %s", path)


def load_vector_store(embedding_model, save_path=VECTOR_STORE_DIR):
    """Load a previously saved, trusted local FAISS index."""
    path = Path(save_path)

    index_file = path / "index.faiss"
    metadata_file = path / "index.pkl"

    if not index_file.is_file() or not metadata_file.is_file():
        raise FileNotFoundError(
            f"No complete FAISS index found at {path}. "
            "Build the index first."
        )

    # FAISS's metadata uses pickle-backed storage. Only load
    # index files created by and trusted by this application.
    return FAISS.load_local(
        str(path),
        embedding_model,
        allow_dangerous_deserialization=True,
    )