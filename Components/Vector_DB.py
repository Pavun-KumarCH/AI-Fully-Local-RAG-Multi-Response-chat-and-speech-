from paths import vector_database_path
from langchain.schema import Document
from langchain_community.vectorstores import Chroma
from langchain.embeddings import OllamaEmbeddings

# Function to setup vector database
def setup_vector_database(text_chunks):
    documents = [Document(page_content=chunk) for chunk in text_chunks]
    vector_db = Chroma.from_documents(
        documents=documents,
        embedding=OllamaEmbeddings(model='nomic-embed-text', show_progress=False),
        persist_directory=vector_database_path,
        collection_name='Local-Rag'
    )
    vector_db.persist()
    return vector_db