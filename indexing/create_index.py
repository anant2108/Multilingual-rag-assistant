from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS



# loader = PyPDFLoader("data/Mahabharata Volume 1.pdf")

documents = []

for file in Path("data").iterdir():

    if file.is_file():
        docs = load_document(str(file))

        documents.extend(docs)

print("Loading PDF...")

documents = loader.load()

print("PDF Loaded")
print("Pages:", len(documents))


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 200
)

chunks = text_splitter.split_documents(documents)

print("Chunks Created")
print("Total Chunks:", len(chunks))


embedding_model = HuggingFaceEmbeddings(
    model_name = "BAAI/bge-small-en-v1.5"
)

# vectors = embedding_model.embed_documents(
#     [chunk.page_content for chunk in chunks]
# )

batch_size = 100

for i in range(0,len(chunks), batch_size):
    batch = chunks[i:i+batch_size]

    if i==0:

        vector_store = FAISS.from_documents(
    batch,
    embedding_model
)
        
    else:

        vector_store.add_documents(batch)

    print(f"Processed {min(i+batch_size,len(chunks))} / {len(chunks)} chunks")

batch_number = (i//batch_size) +1

vector_store.save_local("faiss_index")

print("Done")

# print(type(vector_store))


# print(type(vectors[0]))
# print(len(vectors))
# print(vectors[0])


# print(type(chunks[0]))
# print(len(chunks))
# print(chunks[0])

# print(type(loader))
# print(type(documents))
# print(len(documents))

# print(documents[4])
# print(type(documents[0]))
# # print(documents[0].page_content)
# print(documents[0].metadata)