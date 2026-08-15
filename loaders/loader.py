import os

from langchain_community.document_loaders import(PyPDFLoader, Docx2txtLoader, TextLoader, CSVLoader)

LOADERS = {
    ".pdf" : PyPDFLoader,
    ".docx" : Docx2txtLoader,
    ".txt" : TextLoader,
    ".csv" : CSVLoader,
}

def load_document(file_path):
    extension = os.path.splitext(file_path)[1].lower()

    loader_class = LOADERS.get(extension)

    if loader_class is None:
        raise ValueError(f"Unsupported file type: {extension}")
    
    loader = loader_class(file_path)

    documents = loader.load()

    return documents