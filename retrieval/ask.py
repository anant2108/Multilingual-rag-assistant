from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

embedding_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5"
)

vector_store = FAISS.load_local(
    "faiss_index",
    embedding_model,
    allow_dangerous_deserialization=True
)

question = input("Ask your question: ")

results = vector_store.similarity_search(
    question,
    k=3
    ) 
#The Langchain internally embed the question in the form
# question_vector = embedding_model.embed_query(question) and then writing the above results line

context = ""

for document in results:
    context += document.page_content
    context += "\n\n"

# print(context)


llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash"
)

prompt = f"""
You are a helpful assistant.

Answer the user's question only using the context below.

Context:
{context}

Question:
{question}

"""

response = llm.invoke(prompt)

print(response.content)