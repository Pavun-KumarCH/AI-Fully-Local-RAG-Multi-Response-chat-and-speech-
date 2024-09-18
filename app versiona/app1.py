import streamlit as st
import os
import ollama
from langchain_community.llms import Ollama
from langchain_community.chat_models import ChatOllama
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.document_loaders import PDFPlumberLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.schema import Document
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.embeddings import OllamaEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
 
# Gemini model integration
from langchain_google_genai import GoogleGenerativeAI
from dotenv import load_dotenv, find_dotenv
import google.generativeai as genai

# Import prompt templates from prompt-technique.py
from prompt_techniques_QA import *


# Create a mapping of techniques to their prompt templates
prompt_mapping = {
    "Default": prompt_template,
    # Add other techniques here as needed...
}

# List of models to evaluate
model_names = [
    'gemini-1.5-flash',
    'llama3.1:8b',
    'gemma2:9b',
    'phi3:14b',
    'mistral-nemo:12b',
]

# Paths
pdf_directory = 'data'
vector_database_path = "vector_database"
audio_path = "audio_files"

# Create directories if they don't exist
os.makedirs(pdf_directory, exist_ok=True)
os.makedirs(vector_database_path, exist_ok=True)
os.makedirs(audio_path, exist_ok=True)

# Sidebar for file upload and prompt technique selection
st.title("🤖 AI Enhanced Content Generation Question & Answering System")

with st.sidebar:
    st.title("Upload & Settings")
    uploaded_files = st.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)
    
    # Dropdown for selecting prompt technique
    prompt_technique = st.selectbox("Select Prompt Technique:", list(prompt_mapping.keys()))
    selected_prompt_template = prompt_mapping[prompt_technique]

    # Dropdown for selecting a model (scrollable if list is large)
    selected_model = st.selectbox("Select Model:", model_names)

# Function to process PDFs
def process_pdfs(pdf_files):
    text_chunks = []
    for pdf_file in pdf_files:
        pdf_path = os.path.join(pdf_directory, pdf_file)
        loader = PDFPlumberLoader(pdf_path)
        pages = loader.load_and_split()
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=7500, chunk_overlap=100)
        for page in pages:
            chunks = text_splitter.split_text(page.page_content)
            text_chunks.extend(chunks)  # Flattening the list of chunks
    return text_chunks

# Function to generate embeddings
def generate_embeddings(text_chunks, model_name='nomic-embed-text'):
    embeddings = [ollama.embeddings(model_name, prompt=chunk) for chunk in text_chunks]
    return embeddings

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

# Modified query_model function with additional arguments
def query_model(vector_db, question, selected_prompt_template, selected_model):

    if selected_model == "gemini-1.5-pro":
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.3)
    else:
        llm = ChatOllama(model=selected_model)  # Use default ChatOllama for other models

    QUERY_PROMPT = selected_prompt_template  # Use selected prompt template here
    

    retriever = MultiQueryRetriever.from_llm(
        retriever=vector_db.as_retriever(),
        llm=llm,
        prompt=QUERY_PROMPT
    )

    template = """Answer the question based only on the following context:
    {context}
    Question: {question}"""
    prompt = ChatPromptTemplate.from_template(template)

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    response = chain.invoke(question)
    return response

# Process PDFs if files are uploaded
if uploaded_files:
    with st.spinner("Processing PDFs..."):
        pdf_files = [file.name for file in uploaded_files]
        text_chunks = process_pdfs(pdf_files)
        st.success("PDFs processed successfully.")
        
        with st.spinner("Setting up vector database..."):
            vector_db = setup_vector_database(text_chunks)
        st.success("Vector database set up successfully.")

    # Text input for user question
    question = st.text_input("Ask a question:")

    if question:
        with st.spinner("Getting response from the model..."):
            # Query model using the selected prompt template and model
            response = query_model(vector_db, question, selected_prompt_template, selected_model)
            st.markdown(f"**Answer:**\n{response}")
