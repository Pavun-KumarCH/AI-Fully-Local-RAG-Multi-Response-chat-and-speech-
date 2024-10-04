import os
import streamlit as st
from PyPDF2 import PdfReader 
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Corrected import statement
from langchain_google_genai import GoogleGenerativeAIEmbeddings 
import google.generativeai as genai
from langchain.vectorstores import FAISS  # Corrected import for vector stores
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain  # Corrected import for QA chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv, find_dotenv

# Import prompt templates from prompt-technique.py
from prompt_technique import *

load_dotenv(find_dotenv())

genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))

# Create a mapping of techniques to their prompt templates
prompt_mapping = {
    "Default": prompt_template,
    "Graph of Thought": prompt_template_1,
    "Tree of Thought": prompt_template_2,
    "Graph of Verification": prompt_template_3,
    "Chain-of-Thought (COT)": prompt_template_4,
    "XOT (Everything of Thought)": prompt_template_5,
    "KD-CoT (Knowledge Driven COT)": prompt_template_6,
    "COT-SC (Self-Consistency with COT)": prompt_template_7,
    "Self-Ask": prompt_template_8,
    "Self-Critique": prompt_template_9,
    "Self-Refine": prompt_template_10,
    "Self-Refinement": prompt_template_11,
    "Iterative Prompting": prompt_template_12,
    "Analogical Prompting": prompt_template_13,
    "Input-Output Prompting": prompt_template_14,
    "Least-to-Most Prompting": prompt_template_15,
    "Plan-and-Solve Prompting": prompt_template_16,
    "Sequential Prompting": prompt_template_17,
    "Step-Back Prompting": prompt_template_18,
    "MemPrompt": prompt_template_19,
    "Chain of Density Prompting": prompt_template_20,
    "Reverse JSON Prompting": prompt_template_21,
    "Symbolic Reasoning Prompting": prompt_template_22,
    "Generated Knowledge Prompting": prompt_template_23,
    "PAL (Program-Aided Language Models)": prompt_template_24,
    "Meta-Ask Self-Consistency": prompt_template_25,
    "ReAct": prompt_template_26,
    "ART (Automatic Reasoning & Tool-Use)": prompt_template_27,
    "Few-Shot Prompting": prompt_template_28,
    "Zero-Shot Prompting": prompt_template_29,
    "Chain-of-Thought Prompting": prompt_template_30,
    "Instruction-Based Prompting": prompt_template_31,
    "Persona-Based Prompting": prompt_template_32,
    "Contextual Prompting": prompt_template_33,
    "Role-Playing Prompting": prompt_template_34,
    "Comparison Prompting": prompt_template_35,
    "Multi-Turn Prompting": prompt_template_36,
    "Refinement Prompting": prompt_template_37
}


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")
    return vector_store

def get_conversational_chain(selected_prompt_template):
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3) #gemini-1.5-flash
    prompt = PromptTemplate(template=selected_prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question, selected_prompt_template):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
   
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain(selected_prompt_template)

    response = chain(
        {"input_documents": docs, "question": user_question},
        return_only_outputs=True
    )
    
    print(response)
    st.markdown("**Reply:**")
    st.write(response.get("output_text", "No output generated."))

def main():
    st.set_page_config(page_title="Chat With Agent")
    # Add a banner image
    st.image("assets/langchain.jpg", use_column_width=True)
    st.header("Chat with Intelligent Agent using Gemini💁")

    user_question = st.text_input("Ask a Question from the PDF Files")
    
    # Dropdown for selecting prompt technique
    prompt_technique = st.selectbox("Select Prompt Technique:", list(prompt_mapping.keys()))
    selected_prompt_template = prompt_mapping[prompt_technique]

    if user_question:
        user_input(user_question, selected_prompt_template)

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done")

if __name__ == "__main__":
    main()
