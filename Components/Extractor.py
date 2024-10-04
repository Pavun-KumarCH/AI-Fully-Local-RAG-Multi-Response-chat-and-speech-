import os
from paths import pdf_directory
from langchain_community.document_loaders import PDFPlumberLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

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