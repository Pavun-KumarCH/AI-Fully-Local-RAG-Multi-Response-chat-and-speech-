import streamlit as st
import os
import ollama
import replicate
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

# Import prompt templates from prompt-technique.py
from Prompt_Engineering.prompt_techniques_QA import *

# Create a mapping of techniques to their prompt templates
prompt_mapping = {
    "Default": default_prompt_template,
    "Structured MCQ Generation": mcq_generation_prompt_template,
    "Chain-of-Thought Prompting": chain_of_thought_prompt_template,
    "Simple MCQ Generation": simple_mcq_generation_prompt_template,
    "Chain-of-Thought with Self-Consistency": cot_self_consistency_prompt_template,
    "ReACT Prompting": react_prompt_template,
    "Refined Prompting": refined_prompt_template
}

# List of models to evaluate (includes local and Replicate models)
local_model_names = [
    'llama3.1:8b',
    'gemma2:9b',
    'phi3:14b',
    'mistral-nemo:12b',
]

replicate_model_names = [
    'meta/meta-llama-3-8b',
    'google-deepmind/gemma2-9b-it',
    'microsoft/phi-3-medium-4k-instruct',
    'mistralai/mixtral-8x7b-instruct-v0.1'
]

# Paths
pdf_directory = 'data'
vector_database_path = "vector_database"
audio_path = "audio_files"

# Create directories if they don't exist
os.makedirs(pdf_directory, exist_ok=True)
os.makedirs(vector_database_path, exist_ok=True)
os.makedirs(audio_path, exist_ok=True)

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

# Modified query_model function with added Replicate support
def query_model(vector_db, question, selected_prompt_template, selected_model, model_source):
    if model_source == "Replicate":
        # Use Replicate model
        input = {
            "prompt": f"Context: {question}\nAnswer:",
            "top_p": 0.9,
            "min_tokens": 0,
            "temperature": 0.6,
            "presence_penalty": 1.15
        }
        # Streaming response from replicate
        response = ""
        for event in replicate.stream(
            selected_model,
            input=input
        ):
            response += event['text']
        return response
    
    elif model_source == "Local":
        # Use local Ollama model
        model = ChatOllama(model=selected_model)
        QUERY_PROMPT = selected_prompt_template

        retriever = MultiQueryRetriever.from_llm(
            retriever=vector_db.as_retriever(),
            llm=model,  # Use the correct model
            prompt=QUERY_PROMPT
        )

        # Generate the final response based on the retrieved context
        template = """Answer the question based only on the following context:
        {context}
        Question: {question}"""
        prompt = ChatPromptTemplate.from_template(template)

        chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | model  # Use the correct model
            | StrOutputParser()
        )

        response = chain.invoke(question)
        return response

# Title with emoji
st.title("🤖 AI Enhanced Content Generation Question & Answering System")

# Sidebar for file upload and prompt technique selection
with st.sidebar:
    st.title("Upload & Settings")
    uploaded_files = st.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)
    
    # Dropdown for selecting prompt technique
    prompt_technique = st.selectbox("Select Prompt Technique:", list(prompt_mapping.keys()))
    selected_prompt_template = prompt_mapping[prompt_technique]

    # Dropdown for selecting a model (Replicate and Local)
    model_source = st.radio("Select Model Source:", ("Local", "Replicate"))
    
    if model_source == "Local":
        selected_model = st.selectbox("Select Local Model:", local_model_names)
    else:
        selected_model = st.selectbox("Select Replicate Model:", replicate_model_names)

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
            response = query_model(vector_db, question, selected_prompt_template, selected_model, model_source)
            st.markdown(f"**Answer:**\n{response}")
