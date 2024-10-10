import os
import tempfile
import streamlit as st
from dotenv import load_dotenv, find_dotenv
import google.generativeai as genai

# Import custom modules
from Components import Extractor, Embeddings, Vector_DB, Query_Retriver
from Google_Comp import Extractor_g, Vecotor_dbg, Conversational_chain
from Plagiarism_Checker.checker import plagiarism_check

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

# Helper function to save uploaded files
def save_uploaded_files(uploaded_files):
    temp_dir = tempfile.mkdtemp()
    file_paths = []

    for uploaded_file in uploaded_files:
        file_path = os.path.join(temp_dir, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        file_paths.append(file_path)

    return file_paths

# Helper function to process PDF using Local LLM RAG
def process_with_local_rag(pdf_docs):
    text_chunks = Extractor.process_pdfs(pdf_docs)
    embeddings = Embeddings.generate_embeddings(text_chunks)
    vector_db = Vector_DB.setup_vector_database(text_chunks)
    return vector_db

# Helper function to process PDF using Google Generative AI
def process_with_google(pdf_docs):
    text = Extractor_g.get_pdf_text(pdf_docs)
    text_chunks = Extractor_g.get_text_chunks(text)
    vector_store = Vecotor_dbg.get_vector_store(text_chunks)
    return vector_store

def main():
    st.set_page_config(page_title="AI Agent", layout="wide", initial_sidebar_state="expanded")
    image_url = r"https://static.vecteezy.com/system/resources/previews/008/171/873/non_2x/abstract-boxes-background-modern-technology-with-square-mesh-geometric-on-white-background-with-lines-cube-cell-illustration-free-vector.jpg"

    st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url('{image_url}');
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
        background-position: center;
        color: #333333; /* Darker text color for main content */
        font-family: 'Arial', sans-serif;
    }}

    .overlay {{
        background-color: rgba(0, 0, 0, 0.8); /* Increased overlay opacity */
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        z-index: 1; /* Ensure overlay is on top of the background */
    }}

    h1, h2, h3, h4, h5, h6, p, span {{
        color: #000000; /* Light gray for headings and paragraphs */
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.5); /* Add shadow to text */
    }}
    </style>
    <div class="overlay"></div> <!-- Overlay div -->
    """,
    unsafe_allow_html=True
    )

    if "generated" not in st.session_state:
        st.session_state['generated'] = []
    if "plagiarism" not in st.session_state:
        st.session_state['plagiarism'] = []
    if 'messages' not in st.session_state:
        st.session_state['messages'] = []
    if 'response' not in st.session_state:
        st.session_state['response'] = None
    if 'vector_db' not in st.session_state:
        st.session_state['vector_db'] = None
    if 'vector_store' not in st.session_state:
        st.session_state['vector_store'] = None
    if 'google_conversation_history' not in st.session_state:
        st.session_state['google_conversation_history'] = []

    # Navigation Menu
    menu = ["🏠 Home", "🤖 Chatbot", "📝 Plagiarism", "📧 Contact"]
    choice = st.selectbox("Navigate", menu)

    # Home Page
    if choice == "🏠 Home":
        st.title("🚀 Multi-Model & Multi-Prompt RAG Agent")
        st.markdown("""        
        Welcome to the **Multi-Model & Multi-Prompt RAG Agent**! 😊

        **This application is a powerful Multi-Model and Multi-Prompt Retrieval-Augmented Generation (RAG) Agent.**

        **Utilizing a diverse set of models including:**
    
        - **Gemini 1.5-pro**
        - **Llama 3.1 - 8 Billion**
        - **Llama 3.2 - 3 Billion**
        - **Gemma2 - 9 Billion**
        - **Phi3 - 14 Billion**
        - **Mistral-Nemo - 12 Billion**
        - **Nomic-Embed-Text**

        **Key Features:**
        
        - **Upload Documents:** Seamlessly upload your PDF documents for processing.
        - **Summarize:** Obtain concise and accurate summaries of your documents.
        - **Multi-Prompts:** Dynamically mapped different prompt techniques for tailored responses.
        - **Chat:** Engage with your documents through our intelligent, multi-model chatbot.
        - **Plagiarism Check:** Ensure the originality of your content.

        **Built with an Open Source Stack:**
        
        - **Langchain & Ollama Frameworks**
        - **Google Generative AI**
        - **Qdrant & ChromaDB**  (running locally within a Docker Container)
        - **Nomic-Embed for embeddings**

        Enhance your document management and interaction experience with our RAG Agent! 🚀
        """)

    # Chatbot Page
    elif choice == "🤖 Chatbot":
        st.title("Interact with an Adaptive Multi-Model Multi-Prompt Intelligent Assistant 💁")
        st.markdown("----")
        st.image("assets/langchain.jpg", use_column_width=False)
        st.header("Ask Your Question")

        with st.sidebar:
            st.image("assets/logo.gif", use_column_width=True)
            st.markdown("### 📚 Your Personal Document Assistant")
            st.markdown("----")
            st.subheader("Menu:")
            system_choice = st.radio("**Select System:**", ["**Local LLM RAG**", "**Gemini**"])

            # Local models to evaluate
            local_model_names = [
                'llama3.2',
                'llama3.1:8b',
                'gemma2:9b',
                'phi3:14b',
                'mistral-nemo:12b',
            ]

            if system_choice == "**Local LLM RAG**":
                selected_model = st.selectbox("**Select Local Model:**", local_model_names)
            else:
                selected_model = "**Gemini**"

            pdf_docs = st.file_uploader("Upload PDF Documents", type=['pdf'], accept_multiple_files=True)

            if st.button("Process PDF Files"):
                if pdf_docs:
                    if system_choice == "**Local LLM RAG**":
                        file_paths = save_uploaded_files(pdf_docs)
                        st.session_state.vector_db = process_with_local_rag(file_paths)
                        st.success("PDF files processed and stored successfully!")
                    else:
                        file_paths = save_uploaded_files(pdf_docs)
                        st.session_state.vector_store = process_with_google(file_paths)
                        st.success("PDF files processed with Google Generative AI successfully!")
                else:
                    st.warning("Please upload PDF files first!")

        user_question = st.text_input("Ask a Question from the PDF Files", key="user_question", label_visibility="collapsed")

        if system_choice == "**Local LLM RAG**":
            prompt_technique = st.selectbox("Select Prompt Technique:", list(local_prompt_mapping.keys()), key="local_prompt")
            selected_prompt_template = local_prompt_mapping[prompt_technique]
        else:
            prompt_technique = st.selectbox("Select Prompt Technique:", list(gemini_prompt_mapping.keys()), key="gemini_prompt")
            selected_prompt_template = gemini_prompt_mapping[prompt_technique]

        if user_question:
            if system_choice == "**Local LLM RAG**" and st.session_state.vector_db:
                st.write(f"Processing your question using {system_choice} with the '{prompt_technique}' prompt technique...")
                try:
                    response = Query_Retriver.query_model(
                        st.session_state.vector_db, 
                        user_question, 
                        selected_prompt_template, 
                        selected_model
                    )
                    st.session_state.response = response
                    st.write(f"**Response:** {response}")
                except Exception as e:
                    st.error(f"Error during query processing: {e}")

            elif system_choice == "**Gemini**" and st.session_state.vector_store:
                st.write(f"Processing your question using {system_choice} with the '{prompt_technique}' prompt technique...")
                try:
                    response = Conversational_chain.user_input(user_question, selected_prompt_template)
                    st.session_state.response = response
                    st.session_state.google_conversation_history.append((user_question, response))
                    st.write(f"**Response:** {response}")
                except Exception as e:
                    st.error(f"Error during conversational chain processing: {e}")

        if st.session_state.google_conversation_history:
            st.markdown("### Conversation History")
            for question, answer in st.session_state.google_conversation_history:
                st.markdown(f"**User:** {question}")
                st.markdown(f"**Agent:** {answer}")

        elif st.session_state.response:
            st.write(f"**Response:** {st.session_state.response}")

    # Plagiarism Check Page
    elif choice == "📝 Plagiarism":
        st.title("📝 Plagiarism Checker")
        st.markdown("""
        Paste your content below to check for plagiarism:
        """)

        input_text = st.text_area("Input Text", height=300)

        if st.button("Check Plagiarism"):
            if input_text.strip() == "":
                st.warning("Please paste some text to check for plagiarism.")
            else:
                with st.spinner("Checking for plagiarism..."):
                    try:
                        plagiarism_percentage, sources = plagiarism_check(input_text)
                        if plagiarism_percentage == 0:
                            st.success("No plagiarism detected.")
                        else:
                            st.error(f"**Total Plagiarism Detected:** {plagiarism_percentage:.2f}%")
                            st.write("**Matching Sources and Percentages:**")
                            for url, percentage in sources.items():
                                st.write(f"- [{url}]({url}): {percentage:.2f}%")
                    except Exception as e:
                        st.error(f"Error during plagiarism check: {e}")

    # Contact Page
    elif choice == "📧 Contact":
        st.title("📬 Contact Us")
        st.markdown("""
        We'd love to hear from you! Whether you have a question, feedback, or want to contribute, feel free to reach out.

        - **Email:** [pavun9848@gmail.com](mailto:pavun9848@gmail.com) ✉️
        - **GitHub:** [Contribute on GitHub](https://github.com/Pavun-KumarCH) 🛠️

        If you'd like to request a feature or report a bug, please open a pull request on our GitHub repository. Your contributions are highly appreciated! 🙌
        """)

    # Footer
    st.markdown("---")
    st.markdown("© 2024 Multi-Model & Multi-Prompt RAG Agent by Team-206. All rights reserved. 🛡️")

# Run the main function
if __name__ == "__main__":
    main()
