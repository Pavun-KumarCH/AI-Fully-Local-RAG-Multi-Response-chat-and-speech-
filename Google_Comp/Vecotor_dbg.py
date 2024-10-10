# Vector
from langchain_google_genai import GoogleGenerativeAIEmbeddings 
from langchain.vectorstores import FAISS  # Corrected import for vector stores
# Import prompt templates from prompt-technique.py

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")
    return vector_store
