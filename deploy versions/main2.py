import os
import nltk
import tempfile  # Make sure to import tempfile
import streamlit as st
from Components import Extractor, Embeddings, Vector_DB, Query_Retriver
from Google_Comp import Extractor_g, Vecotor_dbg, Conversational_chain
from Plagiarism_Checker.checker import plagiarism_check  # Import the plagiarism checking function
import google.generativeai as genai
from Plagiarism_Checker import ConsineSim, webSearch
from dotenv import load_dotenv, find_dotenv

# Import prompt templates from prompt-technique.py
from Prompt_Engineering.prompt_techniques import *
from Prompt_Engineering.prompt_techniques_QA import *

# Load environment variables for Google API Key
load_dotenv(find_dotenv())
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))

# Create a mapping of techniques to their prompt templates for Local LLM RAG
local_prompt_mapping = {
    "Default": default_prompt_template,
    "Structured MCQ Generation": mcq_generation_prompt_template,
    "Chain-of-Thought Prompting": chain_of_thought_prompt_template,
    "Simple MCQ Generation": simple_mcq_generation_prompt_template,
    "Chain-of-Thought with Self-Consistency": cot_self_consistency_prompt_template,
    "ReACT Prompting": react_prompt_template,
    "Refined Prompting": refined_prompt_template,
}

# Create a mapping of techniques to their prompt templates for Gemini
gemini_prompt_mapping = {
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
    "Refinement Prompting": prompt_template_37,
}

# Helper function to process PDF using Local LLM RAG
def process_with_local_rag(pdf_docs):
    # Process the PDFs to extract text chunks
    text_chunks = Extractor.process_pdfs(pdf_docs)
    
    # Generate embeddings for the text chunks
    embeddings = Embeddings.generate_embeddings(text_chunks)
    
    # Setup the vector database with the text chunks
    vector_db = Vector_DB.setup_vector_database(text_chunks)
    
    return vector_db, embeddings

# Helper function to process PDF using Google Generative AI
def process_with_google(pdf_docs):
    # Process the PDFs to extract text chunks
    text = Extractor_g.get_pdf_text(pdf_docs)
    text_chunks = Extractor_g.get_text_chunks(text)
    
    # Generate vector store
    vector_store = Vecotor_dbg.get_vector_store(text_chunks)
    
    return vector_store

# Other imports remain the same
# ...

import os
import tempfile
import streamlit as st

def save_uploaded_files(uploaded_files):
    # Create a temporary directory to store uploaded files
    temp_dir = tempfile.mkdtemp()

    # List to store file paths
    file_paths = []

    for uploaded_file in uploaded_files:
        # Create a file path in the temporary directory
        file_path = os.path.join(temp_dir, uploaded_file.name)
        # Save the uploaded file to the specified path
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        file_paths.append(file_path)

    return file_paths, temp_dir


def main():
    st.set_page_config(page_title="Chat with AI Agent")

    # Display banner image
    st.image("assets/langchain.jpg", use_column_width=True)
    st.header("Interact with an Adaptive Multi-Model Multi-Prompt Intelligent Assistant 💁")

    # Initialize session state variables if not already present
    if 'response' not in st.session_state:
        st.session_state.response = None

    # Sidebar for PDF upload and options
    with st.sidebar:
        st.title("Menu:")
        system_choice = st.radio("Select System:", ["Local LLM RAG", "Gemini"])

        # List of local models to evaluate
        local_model_names = [
            'llama3.1:8b',
            'gemma2:9b',
            'phi3:14b',
            'mistral-nemo:12b',
        ]

        if system_choice == "Local LLM RAG":
            selected_model = st.selectbox("Select Local Model:", local_model_names)
        else:
            selected_model = "gemini"

        pdf_docs = st.file_uploader("Upload your PDF Files and Click Submit & Process", accept_multiple_files=True)

        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                if pdf_docs:
                    file_paths, temp_dir = save_uploaded_files(pdf_docs)
                    if system_choice == "Local LLM RAG":
                        vector_db, embeddings = process_with_local_rag(file_paths)
                        st.success("Local LLM RAG processing completed!")
                    else:
                        vector_store = process_with_google(file_paths)
                        st.success("Google processing completed!")
                else:
                    st.warning("Please upload at least one PDF file.")

    # Center the question input directly
    user_question = st.text_input("Ask a Question from the PDF Files", key="user_question", label_visibility="collapsed")

    # Align the prompt technique selection to the right side
    if system_choice == "Local LLM RAG":
        prompt_technique = st.selectbox("Select Prompt Technique:", list(local_prompt_mapping.keys()), key="local_prompt")
        selected_prompt_template = local_prompt_mapping[prompt_technique]
    else:
        prompt_technique = st.selectbox("Select Prompt Technique:", list(gemini_prompt_mapping.keys()), key="gemini_prompt")
        selected_prompt_template = gemini_prompt_mapping[prompt_technique]

    # Once the user asks a question
    if user_question:
        if system_choice == "Local LLM RAG":
            if 'vector_db' in locals():
                st.write(f"Processing your question using {system_choice} with the '{prompt_technique}' prompt technique...")
                response = Query_Retriver.query_model(vector_db, user_question, selected_prompt_template, selected_model)
                st.session_state.response = response  # Store response in session state
                st.write(f"Response: {response}")
            else:
                st.warning("Please process the PDFs first!")
        else:
            if 'vector_store' in locals():
                st.write(f"Processing your question using {system_choice} with the '{prompt_technique}' prompt technique...")
                response = Conversational_chain.user_input(user_question, selected_prompt_template)
                st.session_state.response = response  # Store response in session state
                st.write(f"Response: {response}")
            else:
                st.warning("Please process the PDFs first!")
    elif st.session_state.response:
        st.write(f"Response: {st.session_state.response}")  # Display stored response

    # Plagiarism check section
    st.title("Plagiarism Checker")
    st.write("Paste your content below to check for plagiarism:")

    input_text = st.text_area("Input Text", height=300)

    if st.button("Check Plagiarism"):
        if input_text.strip() == "":
            st.warning("Please paste some text to check for plagiarism.")
        else:
            plagiarism_percentage, sources = plagiarism_check(input_text)
            if plagiarism_percentage == 0:
                st.success("No plagiarism detected.")
            else:
                st.error(f"Total Plagiarism Detected: {plagiarism_percentage:.2f}%")
                st.write("Matching Sources and Percentages:")
                for url, percentage in sources.items():
                    st.write(f"{url}: {percentage:.2f}%")

if __name__ == "__main__":
    main()
