from pathlib import Path

from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
    CSVLoader,
)


# Maps each supported file extension to its loader.
LOADERS = {
    ".pdf": PyPDFLoader,
    ".docx": Docx2txtLoader,
    ".txt": TextLoader,
    ".csv": CSVLoader,
}


def load_document(file_path: str | Path):
    """Load one supported document and preserve its source metadata."""

    path = Path(file_path)

    if not path.is_file():
        raise FileNotFoundError(f"Document not found: {path}")

    loader_class = LOADERS.get(path.suffix.lower())

    if loader_class is None:
        raise ValueError(
            f"Unsupported file type: {path.suffix or '(no extension)'}"
        )

    documents = loader_class(str(path)).load()

    for document in documents:
        document.metadata.setdefault("source", str(path))
        document.metadata.setdefault("file_name", path.name)

    return documents


def load_documents(folder_path: str | Path):
    """Load all supported documents directly inside a folder."""

    folder = Path(folder_path)

    if not folder.is_dir():
        raise NotADirectoryError(
            f"Data folder not found: {folder}"
        )

    files = sorted(
        path
        for path in folder.iterdir()
        if path.is_file()
        and not path.name.startswith(".")
        and path.suffix.lower() in LOADERS
    )

    if not files:
        raise ValueError(
            f"No supported documents found in {folder}"
        )

    documents = []

    for path in files:
        loaded_documents = load_document(path)
        documents.extend(loaded_documents)

    return documents
