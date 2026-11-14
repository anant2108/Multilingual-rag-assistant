from pathlib import Path

# Project root
BASE_DIR = Path(__file__).resolve().parent

# Input documents
DATA_DIR = BASE_DIR / "data"

# Use a fresh path so any older FAISS index remains untouched.
VECTOR_STORE_DIR = BASE_DIR / "faiss_index_v2"

# Embedding configuration
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"

# Chunking configuration (measured in characters)
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Number of chunks embedded in each batch
BATCH_SIZE = 100
