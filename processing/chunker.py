
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import CHUNK_SIZE, CHUNK_OVERLAP


def split_documents(documents):
    """Split loaded documents into smaller chunks."""

    if not documents:
        raise ValueError("Cannot create chunks from empty documents.")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    chunks = text_splitter.split_documents(documents)

    if not chunks:
        raise ValueError(
            "No chunks were created. Check the extracted document text."
        )

    return chunks
